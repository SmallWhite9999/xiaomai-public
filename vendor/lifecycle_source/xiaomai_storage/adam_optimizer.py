"""Named, multi-group Python Adam/AdamW with explicit moments and commit checks.

All optimizer equations use Python scalars with declared FP32/FP64 rounding.
No torch.optim update or native tensor arithmetic computes this update. Native
Tensor/storage/version binding remains, with the existing Python copy backend.
This is an opt-in diagnostic optimizer, not a silent replacement of True's
learner, AMP, scheduler, hooks, distributed optimizer, or CUDA semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import torch
from . import kernels as k, numeric_words as w
from .backend import StorageBackend
from .python_optimizer import values, tensor, sha

FORMAT = 'XIAOMAI_PYTHON_ADAM_V1'
OPTIONS = frozenset(('lr', 'betas', 'eps', 'weight_decay', 'amsgrad',
                     'maximize', 'decoupled_weight_decay'))

class AdamError(RuntimeError):
    pass


def builtin_number(v):
    return type(v) is int or type(v) is float


def validate_options(o):
    if type(o) is not dict or any(type(x) is not str for x in o) or set(o) != OPTIONS:
        raise AdamError('ADAM_OPTIONS_SCHEMA')
    for name in ('lr', 'eps', 'weight_decay'):
        v = o[name]
        if not builtin_number(v) or not math.isfinite(v) or v < 0:
            raise AdamError('FINITE_NONNEGATIVE_OPTION:' + name)
    if type(o['betas']) is not tuple or len(o['betas']) != 2:
        raise AdamError('BETAS_TUPLE_REQUIRED')
    for v in o['betas']:
        if not builtin_number(v) or not math.isfinite(v) or not 0 <= v < 1:
            raise AdamError('BETAS_RANGE')
    for name in ('amsgrad', 'maximize', 'decoupled_weight_decay'):
        if type(o[name]) is not bool:
            raise AdamError('BOOL_OPTION:' + name)
    return dict(o)


def config_bits(o):
    """Do not let Python's -0 == +0 hide a configuration change."""
    validate_options(o)
    return tuple((name, tuple(float(v).hex() for v in o[name]) if name == 'betas'
                  else o[name] if type(o[name]) is bool else float(o[name]).hex())
                 for name in sorted(o))


def clone_data(t):
    with torch.no_grad():
        return k.clone(t, memory_format=torch.contiguous_format)


def descriptor(p):
    k.admit(p)
    if type(p) is not torch.nn.Parameter or not p.is_leaf or p.is_inference() or not p.is_contiguous():
        raise AdamError('ORDINARY_CONTIGUOUS_PARAMETER_REQUIRED')
    if p.dtype is not torch.float32 and p.dtype is not torch.float64:
        raise AdamError('CPU_FP32_FP64_PARAMETER_REQUIRED')
    return (id(p), tuple(p.shape), tuple(p.stride()), str(p.dtype),
            p.storage_offset(), p.untyped_storage()._cdata)


def plain_meta(p):
    return {'shape': list(p.shape), 'dtype': str(p.dtype)}


def tensor_fingerprint(t):
    k.admit(t)
    return (id(t), tuple(t.shape), tuple(t.stride()), t.storage_offset(),
            str(t.dtype), t.untyped_storage()._cdata, t._version, sha(t))


def rounded(value, dtype):
    if not math.isfinite(value):
        raise AdamError('NONFINITE_INTERMEDIATE_NOT_REPAIRED')
    v = w.decode(w.encode_scalar(value, dtype), dtype)
    if not math.isfinite(v):
        raise AdamError('NONFINITE_ROUNDED_VALUE_NOT_REPAIRED')
    return v


class PythonAdam:
    """Constructor takes named parameters or explicit groups of named parameters.

    Parameter identities and group membership cannot be changed after construction.
    Scalar hyperparameters can be edited between steps through param_groups. Step
    counts belong to individual parameters; grad=None skips decay and state update.
    State import/load never silently associates equal-shaped but differently named
    parameters. No higher-order differentiation or captured/fused execution.
    """
    def __init__(self, params, *, lr=1e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=0., amsgrad=False, maximize=False,
                 decoupled_weight_decay=False, sink=None, record_values=True,
                 max_state_elements=None):
        self.defaults = validate_options(dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize,
            decoupled_weight_decay=decoupled_weight_decay))
        if type(record_values) is not bool:
            raise AdamError('BOOL_RECORD_VALUES_REQUIRED')
        if max_state_elements is not None and (type(max_state_elements) is not int or max_state_elements <= 0):
            raise AdamError('EXPLICIT_POSITIVE_RESOURCE_BOUND_REQUIRED')
        self.sink, self.record_values = sink, record_values
        self.max_state_elements = max_state_elements
        self.failed, self.step_calls, self.last_receipt = False, 0, None
        self.param_groups, self.named, self.state = [], {}, {}
        self._lock = threading.RLock()
        source = list(params)
        if not source:
            raise AdamError('NONEMPTY_PARAMETERS_REQUIRED')
        groups = source if type(source[0]) is dict else [{'params': source}]
        for gi, original in enumerate(groups):
            if type(original) is not dict or any(type(x) is not str for x in original):
                raise AdamError('PLAIN_GROUP_REQUIRED')
            if set(original) - (OPTIONS | {'params', 'group_name'}):
                raise AdamError('UNSUPPORTED_GROUP_OPTION')
            options = validate_options({**self.defaults, **{x: v for x, v in original.items() if x in OPTIONS}})
            group_name = original.get('group_name', f'group{gi}')
            if type(group_name) is not str or not group_name:
                raise AdamError('GROUP_NAME_REQUIRED')
            entries = list(original['params'])
            if not entries:
                raise AdamError('EMPTY_GROUP_NOT_ADMITTED')
            names, actual = [], []
            for entry in entries:
                if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str or not entry[0]:
                    raise AdamError('NAMED_PARAMETERS_REQUIRED')
                name, p = entry
                if name in self.named or any(p is old for old in self.named.values()):
                    raise AdamError('DUPLICATE_PARAMETER_NAME_OR_OBJECT')
                descriptor(p)
                self.named[name] = p
                names.append(name); actual.append(p)
            self.param_groups.append(dict(options, params=actual, param_names=names, group_name=group_name))
        if len({g['group_name'] for g in self.param_groups}) != len(self.param_groups):
            raise AdamError('DUPLICATE_GROUP_NAME')
        self.params = list(self.named.values())
        self._bindings = {n: descriptor(p) for n, p in self.named.items()}
        self._group_names = tuple((g['group_name'], tuple(g['param_names'])) for g in self.param_groups)
        stores = [p.untyped_storage()._cdata for p in self.params]
        if len(stores) != len(set(stores)):
            raise AdamError('TRAINABLE_PARAMETER_STORAGE_ALIAS_NOT_ADMITTED')
        self._source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def emit(self, kind, **fields):
        if self.sink is not None:
            self.sink(kind, **fields)

    def _validate(self, *, moments=True):
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._source_sha:
            raise AdamError('OPTIMIZER_IMPLEMENTATION_CHANGED')
        if type(self.param_groups) is not list or len(self.param_groups) != len(self._group_names):
            raise AdamError('GROUP_MEMBERSHIP_CHANGED')
        seen = []
        for group, (group_name, names) in zip(self.param_groups, self._group_names):
            if type(group) is not dict or set(group) != OPTIONS | {'params', 'param_names', 'group_name'}:
                raise AdamError('GROUP_SCHEMA_CHANGED')
            if type(group['group_name']) is not str or group['group_name'] != group_name:
                raise AdamError('GROUP_IDENTITY_CHANGED')
            if type(group['param_names']) is not list or any(type(x) is not str for x in group['param_names']) or tuple(group['param_names']) != names:
                raise AdamError('NAMED_GROUP_BINDING_CHANGED')
            if type(group['params']) is not list or len(group['params']) != len(names):
                raise AdamError('GROUP_MEMBERSHIP_CHANGED')
            for name, p in zip(names, group['params']):
                if p is not self.named[name] or descriptor(p) != self._bindings[name]:
                    raise AdamError('PARAMETER_BINDING_CHANGED:' + name)
                seen.append(p)
            validate_options({n: group[n] for n in OPTIONS})
            if type(self) is PythonAdamW and group['decoupled_weight_decay'] is not True:
                raise AdamError('ADAMW_GROUP_REQUIRES_DECOUPLED_WEIGHT_DECAY')
        if len(seen) != len(self.params) or any(a is not b for a, b in zip(seen, self.params)):
            raise AdamError('PARAMETER_ORDER_CHANGED')
        if type(self.state) is not dict or any(type(n) is not str or n not in self.named for n in self.state):
            raise AdamError('MOMENT_STATE_KEYS')
        if moments:
            stores=set()
            for g in self.param_groups:
                for name in g['param_names']:
                    if name in self.state:
                        self._validate_slots(name, self.state[name], g)
                        for key,t in self.state[name].items():
                            if key!='step':
                                storage=t.untyped_storage()._cdata
                                if storage in stores:raise AdamError('CROSS_PARAMETER_MOMENT_ALIAS')
                                stores.add(storage)

    def _validate_slots(self, name, entry, options):
        expected = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if options['amsgrad'] else set())
        if type(entry) is not dict or set(entry) != expected:
            raise AdamError('MOMENT_STATE_SCHEMA:' + name)
        if type(entry['step']) is not int or not 0 <= entry['step'] <= 2**53 - 1:
            raise AdamError('INTEGER_STEP_OUT_OF_DOMAIN')
        p = self.named[name]
        seen = set(q.untyped_storage()._cdata for q in self.params)
        for key in sorted(expected - {'step'}):
            t = entry[key]; k.admit(t)
            if type(t) is not torch.Tensor or t.requires_grad or t.is_inference() or not t.is_contiguous() or tuple(t.shape) != tuple(p.shape) or t.dtype is not p.dtype:
                raise AdamError('MOMENT_METADATA:' + name + ':' + key)
            store = t.untyped_storage()._cdata
            if store in seen:
                raise AdamError('MOMENT_STORAGE_ALIAS')
            seen.add(store)
            vals = values(t)
            if any(not math.isfinite(x) or (key != 'exp_avg' and x < 0) for x in vals):
                raise AdamError('NONFINITE_OR_NEGATIVE_MOMENT')
        if options['amsgrad'] and any(a < b for a, b in zip(values(entry['max_exp_avg_sq']), values(entry['exp_avg_sq']))):
            raise AdamError('AMSGRAD_MAX_SMALLER_THAN_SECOND_MOMENT')

    def _identity(self):
        self._validate()
        return (self.step_calls,
                tuple(config_bits({n: g[n] for n in OPTIONS}) for g in self.param_groups),
                tuple((n, self._bindings[n], self.state[n]['step'],
                       tuple((key, tensor_fingerprint(value)) for key, value in sorted(self.state[n].items()) if key != 'step'))
                      for n in sorted(self.state)))

    def zero_grad(self, set_to_none=True):
        if type(set_to_none) is not bool:
            raise AdamError('BOOL_ZERO_GRAD_REQUIRED')
        self._validate()
        with torch.no_grad(), StorageBackend(sink=self.sink, record_values=self.record_values):
            for p in self.params:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.zero_()

    def state_dict(self):
        with self._lock:
            self._validate()
            return {'format': FORMAT, 'implementation_sha256': self._source_sha,
                'step_calls': self.step_calls,
                'groups': [{'name': g['group_name'], 'parameters': list(g['param_names']),
                            'options': {n: g[n] for n in OPTIONS}} for g in self.param_groups],
                'parameters': {n: plain_meta(p) for n, p in self.named.items()},
                'state': {n: {key: value if key == 'step' else clone_data(value)
                              for key, value in entry.items()} for n, entry in self.state.items()}}

    def load_state_dict(self, data):
        with self._lock:
            self._validate(moments=False)
            required = {'format', 'implementation_sha256', 'step_calls', 'groups', 'parameters', 'state'}
            if type(data) is not dict or set(data) != required or data['format'] != FORMAT or data['implementation_sha256'] != self._source_sha:
                raise AdamError('STATE_SCHEMA_OR_IMPLEMENTATION_MISMATCH')
            if type(data['step_calls']) is not int or data['step_calls'] < 0:
                raise AdamError('STEP_CALLS_SCHEMA')
            if data['parameters'] != {n: plain_meta(p) for n, p in self.named.items()}:
                raise AdamError('NAMED_PARAMETER_METADATA_MISMATCH')
            if type(data['groups']) is not list or len(data['groups']) != len(self.param_groups):
                raise AdamError('GROUP_STATE_MISMATCH')
            restored_options = []
            for saved, live in zip(data['groups'], self.param_groups):
                if type(saved) is not dict or set(saved) != {'name', 'parameters', 'options'} or saved['name'] != live['group_name'] or saved['parameters'] != live['param_names']:
                    raise AdamError('NAMED_GROUP_ORDER_MISMATCH')
                restored_options.append(validate_options(saved['options']))
            if type(data['state']) is not dict or any(type(n) is not str or n not in self.named for n in data['state']):
                raise AdamError('STATE_PARAMETER_KEYS')
            staged = {}; input_stores=set()
            for options, live in zip(restored_options, self.param_groups):
                for name in live['param_names']:
                    if name in data['state']:
                        self._validate_slots(name, data['state'][name], options)
                        for key,t in data['state'][name].items():
                            if key!='step':
                                storage=t.untyped_storage()._cdata
                                if storage in input_stores:raise AdamError('CROSS_PARAMETER_MOMENT_ALIAS')
                                input_stores.add(storage)
                        staged[name] = {key: value if key == 'step' else clone_data(value)
                                        for key, value in data['state'][name].items()}
            # All data checks precede publishing any option or state change.
            for options, live in zip(restored_options, self.param_groups):
                live.update(options)
            self.state, self.step_calls, self.failed = staged, data['step_calls'], False
            self.last_receipt = {'status': 'STATE_RESTORED', 'native_versions_rewound': False}

    def step(self, closure=None, *, before_commit=None, after_parameter_write=None):
        if closure is not None:
            raise AdamError('CLOSURE_NOT_ADMITTED')
        with self._lock:
            if self.failed:
                raise AdamError('FAILED_OPTIMIZER_REQUIRES_EXPLICIT_RESTORE')
            start = self._identity()
            prepared = []
            commits = []
            old_state, old_step_calls = self.state, self.step_calls
            element_count = 0
            try:
                for gi, group in enumerate(self.param_groups):
                    o = {n: group[n] for n in OPTIONS}
                    b1, b2 = map(float, o['betas'])
                    for name in group['param_names']:
                        p = self.named[name]; g = p.grad
                        if g is None:
                            self.emit('PY_ADAM_SKIPPED', parameter=name, group=gi, reason='GRAD_NONE_NO_DECAY_NO_STEP')
                            continue
                        k.admit(g)
                        if type(g) is not torch.Tensor or g.requires_grad or g.is_inference() or tuple(g.shape) != tuple(p.shape) or g.dtype is not p.dtype:
                            raise AdamError('GRADIENT_METADATA:' + name)
                        if g.untyped_storage()._cdata in {q.untyped_storage()._cdata for q in self.params}:
                            raise AdamError('GRADIENT_PARAMETER_STORAGE_ALIAS')
                        ph, gh = tensor_fingerprint(p), tensor_fingerprint(g)
                        pv, gv = values(p), values(g)
                        element_count += len(pv)
                        if self.max_state_elements is not None and element_count > self.max_state_elements:
                            raise AdamError('EXPLICIT_PREPARATION_ELEMENT_LIMIT')
                        if any(not math.isfinite(x) for x in pv + gv):
                            self.emit('PY_ADAM_NONFINITE_INPUT', parameter=name,
                                parameter_bytes=k.logical_bytes(p).hex(), gradient_bytes=k.logical_bytes(g).hex(), repaired=False)
                            raise AdamError('NONFINITE_INPUT_NOT_REPAIRED:' + name)
                        old = self.state.get(name)
                        step = 1 if old is None else old['step'] + 1
                        if step > 2**53 - 1:
                            raise AdamError('STEP_OUT_OF_DOMAIN')
                        av = (0.,) * len(pv) if old is None else values(old['exp_avg'])
                        vv = (0.,) * len(pv) if old is None else values(old['exp_avg_sq'])
                        vmax = (0.,) * len(pv) if old is None or not o['amsgrad'] else values(old['max_exp_avg_sq'])
                        correction1, correction2 = 1. - b1**step, 1. - b2**step
                        correction2_sqrt = math.sqrt(correction2)
                        if correction1 <= 0 or correction2_sqrt <= 0:
                            raise AdamError('INVALID_BIAS_CORRECTION')
                        step_size = float(o['lr']) / correction1
                        out, next_a, next_v, next_max = [], [], [], []
                        R = lambda x: rounded(x, p.dtype)
                        for j, (param, grad, m, v) in enumerate(zip(pv, gv, av, vv)):
                            effective = -grad if o['maximize'] else grad
                            decayed = param
                            if o['decoupled_weight_decay']:
                                decayed = R(param * (1. - float(o['lr']) * float(o['weight_decay'])))
                            elif o['weight_decay']:
                                effective = R(effective + float(o['weight_decay']) * param)
                            # Mirror single-tensor Adam's arithmetic structure, but
                            # make each rounding boundary inspectable in Python.
                            difference = R(effective - m)
                            new_m = R(m + (1. - b1) * difference)
                            scaled_v = R(b2 * v)
                            new_v = R(scaled_v + (1. - b2) * effective * effective)
                            max_v = max(vmax[j], new_v) if o['amsgrad'] else new_v
                            root = R(math.sqrt(max_v))
                            unbiased_root = R(root / correction2_sqrt)
                            denominator = R(unbiased_root + float(o['eps']))
                            if denominator == 0.:
                                raise AdamError('ZERO_DENOMINATOR_NOT_REPAIRED')
                            after = R(decayed - step_size * new_m / denominator)
                            out.append(after); next_a.append(new_m); next_v.append(new_v)
                            if o['amsgrad']:
                                next_max.append(max_v)
                            if self.record_values:
                                self.emit('PY_ADAM_ELEMENT', parameter=name, group=gi, index=j, step=step,
                                    parameter_before=param.hex(), gradient=grad.hex(), effective_gradient=effective.hex(),
                                    exp_avg_before=m.hex(), exp_avg_after=new_m.hex(), exp_avg_sq_before=v.hex(),
                                    exp_avg_sq_after=new_v.hex(), max_exp_avg_sq=max_v.hex() if o['amsgrad'] else None,
                                    bias_correction1=correction1.hex(), bias_correction2=correction2.hex(),
                                    denominator=denominator.hex(), decayed_parameter=decayed.hex(),
                                    parameter_after=after.hex(), dtype=str(p.dtype),
                                    parameter_before_bits=w.encode_scalar(param, p.dtype).hex(),
                                    parameter_after_bits=w.encode_scalar(after, p.dtype).hex())
                        slots = {'step': step, 'exp_avg': tensor(tuple(p.shape), next_a, p.dtype),
                                 'exp_avg_sq': tensor(tuple(p.shape), next_v, p.dtype)}
                        if o['amsgrad']:
                            slots['max_exp_avg_sq'] = tensor(tuple(p.shape), next_max, p.dtype)
                        target, original = tensor(tuple(p.shape), out, p.dtype), clone_data(p)
                        prepared.append(dict(name=name, parameter=p, gradient=g, ph=ph, gh=gh,
                            target=target, target_sha=sha(target), original=original, original_sha=sha(original),
                            slots=slots, slots_sha={n: sha(t) for n, t in slots.items() if n != 'step'}))
                self.emit('PY_ADAM_PREPARED', step_call=self.step_calls + 1,
                          parameters=len(prepared), elements=element_count, native_optimizer=False)
                if before_commit is not None:
                    before_commit()
                self._check_prepared(start, prepared)
                with torch.no_grad(), StorageBackend(sink=self.sink, record_values=self.record_values):
                    for row in prepared:
                        p, g = row['parameter'], row['gradient']
                        if p.grad is not g or tensor_fingerprint(p) != row['ph'] or tensor_fingerprint(g) != row['gh']:
                            raise AdamError('SOURCE_CHANGED_DURING_COMMIT')
                        # Include the current parameter before attempting write;
                        # a failing writer can already have changed a partial buffer.
                        commits.append(row)
                        p.copy_(row['target'])
                        if after_parameter_write is not None:
                            after_parameter_write(row['name'], p)
                        if sha(p) != row['target_sha']:
                            raise AdamError('PARAMETER_WRITE_READBACK_MISMATCH')
                # A later callback may have changed an earlier parameter.
                for row in prepared:
                    if sha(row['parameter']) != row['target_sha']:
                        raise AdamError('POST_COMMIT_PARAMETER_MISMATCH')
                if self._identity() != start:
                    raise AdamError('OPTIMIZER_STATE_CHANGED_DURING_COMMIT')
                new_state = dict(old_state)
                new_state.update({r['name']: r['slots'] for r in prepared})
                self.state, self.step_calls = new_state, old_step_calls + 1
                self.last_receipt = {'status': 'UPDATE_COMMITTED', 'parameters': len(prepared),
                                     'elements': element_count, 'step_call': self.step_calls,
                                     'native_optimizer': False}
                final_identity=self._identity()
                self.emit('PY_ADAM_COMMITTED', **self.last_receipt)
                if self._identity()!=final_identity:raise AdamError('OBSERVER_CHANGED_MOMENTS_OR_OPTIONS_AFTER_COMMIT')
                # Detect observational side effects after the successful event too.
                for row in prepared:
                    if sha(row['parameter']) != row['target_sha']:
                        raise AdamError('OBSERVER_CHANGED_PARAMETER_AFTER_COMMIT')
                return None
            except BaseException as error:
                restored = False
                if commits:
                    try:
                        with torch.no_grad(), StorageBackend():
                            for row in reversed(commits):
                                row['parameter'].copy_(row['original'])
                        restored = all(sha(row['parameter']) == row['original_sha'] for row in commits)
                    except BaseException:
                        restored = False
                self.state, self.step_calls = old_state, old_step_calls
                self.failed = True
                self.last_receipt = {'status': 'UPDATE_FAILED', 'writes_attempted': len(commits),
                    'written_parameter_bytes_restored': restored if commits else None,
                    'native_version_counters_restored': False, 'explicit_restore_required': True,
                    'error_type': type(error).__name__}
                # Do not replace the actual exception if the event sink itself failed.
                try:
                    self.emit('PY_ADAM_FAILED', **self.last_receipt)
                except BaseException:
                    pass
                raise

    def _check_prepared(self, start, prepared):
        if self._identity() != start:
            raise AdamError('OPTIMIZER_STATE_CHANGED_BEFORE_COMMIT')
        for row in prepared:
            p, g = row['parameter'], row['gradient']
            if p.grad is not g or tensor_fingerprint(p) != row['ph'] or tensor_fingerprint(g) != row['gh']:
                raise AdamError('SOURCE_CHANGED_BEFORE_COMMIT')
            if sha(row['target']) != row['target_sha'] or sha(row['original']) != row['original_sha']:
                raise AdamError('STAGED_UPDATE_CHANGED')
            if any(sha(row['slots'][n]) != h for n, h in row['slots_sha'].items()):
                raise AdamError('STAGED_MOMENT_CHANGED')


class PythonAdamW(PythonAdam):
    def __init__(self, params, *, weight_decay=.01, **kwargs):
        if kwargs.get('decoupled_weight_decay', True) is not True:
            raise AdamError('ADAMW_REQUIRES_DECOUPLED_WEIGHT_DECAY')
        kwargs['decoupled_weight_decay'] = True
        super().__init__(params, weight_decay=weight_decay, **kwargs)
        if any(g['decoupled_weight_decay'] is not True for g in self.param_groups):
            raise AdamError('ADAMW_GROUP_REQUIRES_DECOUPLED_WEIGHT_DECAY')
