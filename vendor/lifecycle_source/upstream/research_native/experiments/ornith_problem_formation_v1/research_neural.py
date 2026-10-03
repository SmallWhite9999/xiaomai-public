"""Registered research state and an explicit decoder bridge.

Evidence features are the real backbone's final normalized token activations,
mean pooled over each unpadded row, detached, then projected by a trainable
encoder. The encoder never reads a hidden-cause label or a hash feature. Numeric
and type information must be supplied as tokens by the evidence serializer;
this first encoder does not claim a calibrated numerical representation.

Full-prefix recomputation remains the default and the training path. Optional
research-cache-v1 inference owns one disposable native/graft cache with fixed Z;
unowned KV/GDN/conv caches are rejected, not reused across changed Z/weights.
Only token inputs and detached inference state are retained; training rebuilds
Z and its graph within a scoped, re-entrant context. No hooks are installed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math

import torch
from torch import Tensor, nn

from experiments.ornith_research_credit_v1.baseline.research_primitives import (
    ResearchBridge, ResearchStateCore,
)
from .attention_memory import bounded_attention_workspace


class EvidenceEncoder(nn.Module):
    """Trainable projection of actual frozen backbone token features."""

    def __init__(self, backbone_width: int, width: int):
        super().__init__()
        self.projection = nn.Linear(backbone_width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, pooled: Tensor) -> Tensor:
        if pooled.ndim != 2 or pooled.shape[1] != self.projection.in_features:
            raise ValueError('expected pooled token rows [N, backbone_width]')
        pooled = pooled.to(self.projection.weight)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError('nonfinite real backbone features')
        return self.norm(self.projection(pooled))


class ResearchNeuralComponents(nn.Module):
    """One registered owner of encoder, core, outcome/Q heads and bridge."""

    def __init__(self, backbone_width: int, width: int, slots: int,
                 outcome_classes: int, steps: int):
        super().__init__()
        self.encoder = EvidenceEncoder(backbone_width, width)
        self.core = ResearchStateCore(width, slots, outcome_classes)
        self.bridge = ResearchBridge(backbone_width, width)
        # A separate typed target: execution categories must not stand in for
        # the value of a numerical measurement. This head is only trained by
        # an explicitly precommitted numeric observation contract.
        self.outcome_mean = nn.Linear(2 * width, 1)
        self.steps = steps

    def state_from_features(self, evidence: Tensor) -> Tensor:
        if evidence.ndim != 2 or evidence.shape[-1] != self.core.width:
            raise ValueError('evidence features must be [N,width]')
        if evidence.shape[0] == 0:
            # Absence is an explicit initial state, not an all-masked softmax
            # or a fabricated zero measurement.
            return self.core.initial.unsqueeze(0)
        encoded = evidence.unsqueeze(0)
        # The reference core's state recurrence does not depend on candidates.
        # Use a real evidence row for its unused scoring input; only state is
        # consumed here. Drafts never enter the evidence recurrence.
        return self.core(encoded, encoded[:, :1], steps=self.steps)['state']

    def score(self, state: Tensor, candidates: Tensor) -> dict[str, Tensor]:
        if state.ndim != 3 or state.shape[0] != 1:
            raise ValueError('research scoring requires one research stream')
        if candidates.ndim != 2 or candidates.shape[0] < 1:
            raise ValueError('no usable candidates')
        pooled = state.mean(1)
        features = candidates.unsqueeze(0)
        logits = torch.einsum('bd,bad->ba', self.core.action_score(pooled), features)
        paired = torch.cat((pooled[:, None].expand(-1, len(candidates), -1), features), -1)
        return {'state': state,
                'action_logits': logits / math.sqrt(self.core.width),
                'estimated_progress': self.core.value_head(paired).squeeze(-1),
                'outcome_logits': self.core.outcome_head(paired),
                'outcome_mean': self.outcome_mean(paired).squeeze(-1)}


class _Binding:
    """Ephemeral explicit forward context shared by wrapper and runtime."""

    def __init__(self):
        self.stack = []
        self.encode_depth = 0

    def __getstate__(self):
        if self.stack or self.encode_depth:
            raise RuntimeError('RESEARCH_ACTIVE_CONTEXT_NOT_CHECKPOINT_STATE')
        return {'stack': [], 'encode_depth': 0}

    def __deepcopy__(self, memo):
        # A candidate is not executing the original's active computation.
        result = type(self)()
        memo[id(self)] = result
        return result


class ResearchDecoderBlock(nn.Module):
    """An explicit, serializable decoder wrapper, with no module hooks."""

    def __init__(self, native: nn.Module, components: ResearchNeuralComponents,
                 binding: _Binding):
        super().__init__()
        self.native = native
        # Components are registered exactly once on the root model. These are
        # ordinary object-graph references; deepcopy/checkpoint preserves aliases.
        object.__setattr__(self, 'research_components', components)
        self.binding = binding

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            native = super().__getattr__('native')
            return getattr(native, name)

    def forward(self, hidden_states, *args, **kwargs):
        if args:
            raise ValueError('research decoder requires explicit named arguments')
        entry = self.binding.stack[-1] if self.binding.stack and not self.binding.encode_depth else None
        if kwargs.get('use_cache', False) or any(kwargs.get(key) is not None for key in
                ('past_key_values', 'past_key_value', 'cache_params')):
            lease = (getattr(entry['runtime'].model, '_ornith_research_cache_lease', None)
                     if entry is not None else None)
            if lease is None:
                raise ValueError('RESEARCH_CACHE_FORBIDDEN_FULL_PREFIX_RECOMPUTATION_REQUIRED')
            cache = next((kwargs[key] for key in ('past_key_values', 'past_key_value', 'cache_params')
                          if kwargs.get(key) is not None), None)
            lease.authorize_decoder(entry['runtime'], entry, cache)
        # Detect ordinary optimizer updates and Parameter replacement before
        # another token uses a stale context. Explicit .data writes additionally
        # require runtime.mark_updated(); see the installation receipt.
        if entry is not None and (entry['signature'] != entry['runtime']._parameter_signature()
                                  or entry['state_version'] != entry['state']._version):
            raise RuntimeError('RESEARCH_STATE_STALE_DURING_ACTIVE_DECODE')
        result = self.native(hidden_states, **kwargs)
        if entry is None or entry['bypass']:
            return result
        x = result[0] if isinstance(result, tuple) else result
        if not isinstance(x, Tensor):
            raise TypeError('unsupported decoder output contract')
        z = entry['state']
        if x.shape[0] != 1:
            raise ValueError('research bridge currently requires one unpadded stream')
        bridge = self.research_components.bridge
        _, residual = bridge(x.to(bridge.query.weight), z.to(bridge.key.weight))
        # Native stream dtype is retained while the new branch computes in its
        # registered parameter dtype. Casts remain differentiable.
        amended = x + residual.to(x.dtype)
        return (amended, *result[1:]) if isinstance(result, tuple) else amended


def _backbone(model):
    root = getattr(model, 'model', None)
    result = getattr(root, 'language_model', root)
    if result is None or not hasattr(result, 'layers') or not hasattr(result, 'embed_tokens'):
        raise ValueError('supported decoder backbone with layers/embed_tokens required')
    return result


def _rows(rows, vocabulary: int):
    if not isinstance(rows, (list, tuple)):
        raise TypeError('token rows must be a list/tuple')
    result = []
    for row in rows:
        if isinstance(row, Tensor):
            if row.ndim != 1 or row.dtype != torch.long:
                raise ValueError('token tensor rows must be long [T]')
            row = row.detach().cpu().tolist()
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError('each visible token row must be nonempty')
        if any(type(token) is not int or token < 0 or token >= vocabulary for token in row):
            raise ValueError('token ID outside actual backbone vocabulary')
        result.append(list(row))
    return result


class ResearchNeuralRuntime:
    """Event-versioned integration; snapshot training never replays tools.

    set_evidence accepts the complete currently disclosed event frontier. Each
    event ID is immutable within this runtime; snapshots preserve old disclosed
    frontiers for pre-observation loss recomputation. It is the evidence store's
    responsibility to assign truthful IDs, status/type/unit tokens and visibility.
    """

    def __init__(self, model, binding, layer):
        self.model = model
        self.binding = binding
        self.layer = layer
        self.weight_revision = 0
        self._snapshot = {'schema': 'research-neural-input-v1', 'token_rows': [],
                          'event_ids': [], 'concept_version': '0', 'source_version': ''}
        self._event_tokens = {}
        self._inference_state = None
        self._inference_key = None
        self.state_recomputations = 0
        object.__setattr__(model, '_ornith_research_neural_runtime', self)

    @property
    def components(self):
        return self.model.research_neural

    def __getstate__(self):
        if self.binding.stack or self.binding.encode_depth:
            raise RuntimeError('RESEARCH_ACTIVE_CONTEXT_NOT_CHECKPOINT_STATE')
        result = self.__dict__.copy()
        # Recompute after deserialization: process-local parameter IDs and
        # optimizer tensor versions are not portable latent provenance.
        result['_inference_key'] = None
        if self._inference_state is not None:
            result['_inference_state'] = self._inference_state.detach().clone()
        return result

    def __deepcopy__(self, memo):
        # Model deepcopy can happen during a scoped research call. A candidate
        # receives the same detached input frontier, never the active graph or
        # decode lease. memo preserves candidate-model/runtime/wrapper aliases.
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        state = self.__dict__.copy()
        state['_inference_state'] = None
        state['_inference_key'] = None
        result.__dict__.update(copy.deepcopy(state, memo))
        return result

    def _parameter_signature(self):
        wrapper = _backbone(self.model).layers[self.layer]
        if (not isinstance(wrapper, ResearchDecoderBlock) or wrapper.binding is not self.binding
                or wrapper.research_components is not self.components):
            raise RuntimeError('RESEARCH_COMPONENT_BINDING_CHANGED_REQUIRES_EXPLICIT_MIGRATION')
        functions = []
        lease = getattr(self.model, '_ornith_research_cache_lease', None)
        for name, module in self.model.named_modules():
            forward = lease.original_forward(module) if lease is not None else module.forward
            functions.append((name, type(module), id(getattr(forward, '__func__', forward)),
                              id(getattr(forward, '__self__', module))))
        return (self.weight_revision, self.layer, self.components.steps,
                self.components.core.width, self.components.core.slots, tuple(functions),
                tuple((name, id(p), p._version, p.requires_grad)
                      for name, p in self.model.named_parameters()))

    def _validate_snapshot(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('schema') != 'research-neural-input-v1':
            raise ValueError('unsupported research input snapshot')
        rows = _rows(snapshot.get('token_rows'), _backbone(self.model).embed_tokens.num_embeddings)
        ids = snapshot.get('event_ids')
        if not isinstance(ids, (list, tuple)) or len(ids) != len(rows):
            raise ValueError('one event ID per visible evidence row required')
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError('nonempty unique evidence event IDs required')
        for key in ('concept_version', 'source_version'):
            if not isinstance(snapshot.get(key), str):
                raise ValueError('explicit string concept/source versions required')
        return {'schema': 'research-neural-input-v1', 'token_rows': rows,
                'event_ids': list(ids), 'concept_version': snapshot['concept_version'],
                'source_version': snapshot['source_version']}

    def input_snapshot(self):
        return copy.deepcopy(self._snapshot)

    def set_evidence(self, token_rows, event_ids, *, concept_version='0', source_version=''):
        if self.binding.stack or self.binding.encode_depth:
            raise RuntimeError('cannot ingest new evidence during active decode')
        snapshot = self._validate_snapshot({'schema': 'research-neural-input-v1',
            'token_rows': token_rows, 'event_ids': event_ids, 'concept_version': concept_version,
            'source_version': source_version})
        for event, tokens in zip(snapshot['event_ids'], snapshot['token_rows']):
            if event in self._event_tokens and self._event_tokens[event] != tokens:
                raise ValueError('existing evidence event token content is immutable')
        for event, tokens in zip(snapshot['event_ids'], snapshot['token_rows']):
            self._event_tokens[event] = list(tokens)
        if snapshot != self._snapshot:
            self._inference_key = None
            self._inference_state = None
        self._snapshot = snapshot
        return self.version()

    def version(self, snapshot=None):
        snapshot = self._snapshot if snapshot is None else snapshot
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        return {'input_sha256': digest, 'event_index': len(snapshot['event_ids']),
                'weight_revision': self.weight_revision,
                'concept_version': snapshot['concept_version'],
                'source_version': snapshot['source_version']}

    def mark_updated(self):
        self.weight_revision += 1
        self._inference_state = None
        self._inference_key = None

    @contextmanager
    def _encode_scope(self):
        self.binding.encode_depth += 1
        # Feature extraction has a fixed eval-mode interpretation, regardless
        # of downstream trainable projection/core mode. Preserve every module's
        # flag, including heterogeneous custom submodule training settings.
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            yield
        finally:
            for module, flag in modes:
                module.training = flag
            self.binding.encode_depth -= 1

    def encode_only(self, token_rows):
        backbone = _backbone(self.model)
        rows = _rows(token_rows, backbone.embed_tokens.num_embeddings)
        if not rows:
            return self.components.encoder.projection.weight.new_empty(0, self.components.core.width)
        pooled = []
        with self._encode_scope(), torch.no_grad(), bounded_attention_workspace(self.model):
            for row in rows:
                ids = torch.tensor([row], dtype=torch.long, device=backbone.embed_tokens.weight.device)
                output = backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  use_cache=False, return_dict=True)
                features = output.last_hidden_state
                if not bool(torch.isfinite(features).all()):
                    raise ValueError('nonfinite backbone evidence features')
                pooled.append(features.float().mean(1).squeeze(0).detach())
        return self.components.encoder(torch.stack(pooled))

    def _state(self, snapshot, *, training):
        # Even when called inside a generator's inference_mode, keep Z as a
        # normal detached tensor with a mutation version counter. Training must
        # rebuild an ordinary autograd graph rather than an inference tensor.
        with torch.inference_mode(False):
            signature = self._parameter_signature()
            key = (json.dumps(snapshot, sort_keys=True, separators=(',', ':')), signature)
            if not training and self._inference_key == key and self._inference_state is not None:
                return self._inference_state.clone()
            with torch.set_grad_enabled(training):
                state = self.components.state_from_features(self.encode_only(snapshot['token_rows']))
            self.state_recomputations += 1
            if not training:
                state = state.detach()
                self._inference_key, self._inference_state = key, state.clone()
            return state

    @contextmanager
    def use_snapshot(self, snapshot, *, training=False, bypass=False):
        # GroupPolicy calls model(...) directly rather than EOSGenerator, so
        # it does not enter that generator's attention workspace context.
        # Scope the same all-key query-row tiling around every research forward.
        # It preserves the original mask/function and falls back to the original
        # eager implementation for gradients or requested attention matrices.
        with bounded_attention_workspace(self.model):
            snapshot = self._validate_snapshot(snapshot)
            state = self._state(snapshot, training=training)
            entry = {'state': state, 'snapshot': snapshot, 'training': bool(training),
                     'signature': self._parameter_signature(), 'state_version': state._version,
                     'runtime': self, 'bypass': bool(bypass)}
            self.binding.stack.append(entry)
            try:
                yield state
            finally:
                if not self.binding.stack or self.binding.stack[-1] is not entry:
                    raise RuntimeError('research contexts must exit in stack order')
                self.binding.stack.pop()

    def activate(self, *, training=False, bypass=False):
        return self.use_snapshot(self.input_snapshot(), training=training, bypass=bypass)

    def score_candidates(self, token_rows, *, training=False):
        if not token_rows:
            raise ValueError('NO_USABLE_RESEARCH_CANDIDATES')
        if self.binding.stack:
            entry = self.binding.stack[-1]
            if entry['training'] != bool(training):
                raise ValueError('candidate scoring must match active gradient context')
            if (entry['signature'] != self._parameter_signature()
                    or entry['state_version'] != entry['state']._version):
                raise RuntimeError('RESEARCH_STATE_STALE_DURING_ACTIVE_DECODE')
            state = entry['state']
        else:
            state = self._state(self.input_snapshot(), training=training)
        with torch.set_grad_enabled(training):
            return self.components.score(state, self.encode_only(token_rows))

    def installation_receipt(self):
        return {'status': 'INSTALLED', 'profile': 'research-core-v1', 'layer': self.layer,
                'width': self.components.core.width, 'slots': self.components.core.slots,
                'steps': self.components.steps, 'model_child': 'research_neural',
                'encoder': 'final-normalized-backbone-token-mean/trainable-projection',
                'encoder_backbone_gradients': False, 'empty_evidence': 'learned-initial-state',
                'numerics': 'numeric-and-type-content-as-evidence-tokens',
                'numeric_head': 'PRECOMMITTED_TYPED_GAUSSIAN_MEAN_NOT_AUTOMATIC_UNIT_OR_CALIBRATION',
                'cache_policy': 'FULL_PREFIX_RECOMPUTATION_NO_NATIVE_CACHE',
                'optional_inference_cache': 'research-cache-v1: ONE_FIXED_SNAPSHOT_CANDIDATE_NATIVE_AND_GRAFT_STATE',
                'attention_workspace': 'INFERENCE_QUERY_ROWS_128_ALL_KEYS_ORIGINAL_MASK; GRADIENTS_USE_ORIGINAL_EAGER',
                'parameter_versioning': 'tensor-version/identity plus explicit mark_updated for .data writes',
                'state_version': self.version(self.binding.stack[-1]['snapshot']
                                              if self.binding.stack else None),
                'trainable_parameters': [name for name, p in self.model.named_parameters()
                                          if name.startswith('research_neural.') and p.requires_grad]}


def install_research_neural(model, width=32, slots=4, layer=-1, steps=2, outcome_classes=2):
    """Install once without resetting native/graft parameters or changing flags.

    Existing backbone trainability is preserved. New components are FP32 on the
    selected block's device and registered on the actual generating model.
    """
    if hasattr(model, 'research_neural'):
        raise ValueError('research neural already installed; restore its runtime instead of resetting it')
    if any(type(x) is not int or x < 1 for x in (width, slots, steps, outcome_classes)):
        raise ValueError('positive integer research dimensions required')
    if outcome_classes < 2 or steps > 128:
        raise ValueError('reference kernel requires >=2 outcome classes and 1..128 internal steps')
    backbone = _backbone(model)
    if type(layer) is not int or not -len(backbone.layers) <= layer < len(backbone.layers):
        raise ValueError('research layer outside actual decoder')
    layer %= len(backbone.layers)
    native = backbone.layers[layer]
    if isinstance(native, ResearchDecoderBlock):
        raise ValueError('decoder already wrapped')
    parameter = next(native.parameters())
    components = ResearchNeuralComponents(backbone.config.hidden_size, width, slots,
                                          outcome_classes, steps).to(device=parameter.device, dtype=torch.float32)
    binding = _Binding()
    wrapper = ResearchDecoderBlock(native, components, binding)
    model.add_module('research_neural', components)
    backbone.layers[layer] = wrapper
    for config in (getattr(model, 'config', None), getattr(backbone, 'config', None),
                   getattr(getattr(model, 'config', None), 'text_config', None),
                   getattr(model, 'generation_config', None)):
        if config is not None:
            config.use_cache = False
    runtime = ResearchNeuralRuntime(model, binding, layer)
    # Ordinary object reference, not a second registered nn.Module owner. This
    # also lets all EOSGenerator/rescue entrypoints find the same runtime.
    object.__setattr__(model, '_ornith_research_neural_runtime', runtime)
    return runtime
