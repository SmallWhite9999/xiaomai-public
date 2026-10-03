"""A complete *ancestor layer 0* CPU diagnostic, not a restored current True.

Uses all parameters at their configured sizes; no random hidden states, no
weight slicing. The input is retrieved from the admitted real embedding rows.
The original pinned decoder bodies are retained. No-cache layer 0 is GDN, so
there is no invented rotary/cache implementation. Native CPU and scalar-Python
are compared explicitly. No result here is a full-True or GPU acceptance.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import os
import time
import types
from .real_assets import AssetError, REVISION
from .long_paths import write_json_new
from .limits import Budget, Limits, ResourceLimit

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'model.language_model.layers.0.'
EMBEDDING = 'model.language_model.embed_tokens.weight'


def tensor_hash(tensor):
    import torch
    with torch.no_grad(), torch._C._DisableTorchDispatch():
        a = tensor.detach().contiguous().view(torch.uint8).numpy()
        return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


def parameter_hashes(model):
    return {n: tensor_hash(p) for n,p in model.named_parameters()}


COMPARISON_CONTRACT = 'DURABLE_COMPARISONS_V083'


def compare_tensors(output, name, actual, reference, *, relation='bitwise_equal',
                    atol=0.0, rtol=0.0, spec_sha=None, progress=None, phase_seconds=None):
    """Persist one bounded scalar summary before accepting or rejecting a check.

    Full tensor values remain in the trace. This receipt never samples them:
    numerical reductions scan every output element, using bounded chunks.
    """
    import torch
    if relation not in ('allclose','bitwise_equal','bitwise_different'):
        raise ValueError('UNKNOWN_COMPARISON_RELATION')
    if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (atol,rtol)):
        raise ValueError('FINITE_NONNEGATIVE_TOLERANCES_REQUIRED')
    started=time.monotonic()
    same_shape=tuple(actual.shape)==tuple(reference.shape)
    same_dtype=actual.dtype==reference.dtype
    actual_sha=tensor_hash(actual);reference_sha=tensor_hash(reference)
    bitwise=same_shape and same_dtype and actual_sha==reference_sha
    max_abs=0.0;max_excess=0.0;outside=0;nonfinite=0
    if same_shape:
        with torch.no_grad(),torch._C._DisableTorchDispatch():
            left=actual.detach().reshape(-1);right=reference.detach().reshape(-1)
            for begin in range(0,left.numel(),65536):
                if progress and progress.budget:progress.budget.check()
                a=left[begin:begin+65536].to(torch.float64)
                b=right[begin:begin+65536].to(torch.float64)
                finite=torch.isfinite(a)&torch.isfinite(b)
                nonfinite+=int((~finite).sum().item())
                if finite.any():
                    delta=(a[finite]-b[finite]).abs()
                    allowed=atol+rtol*b[finite].abs()
                    max_abs=max(max_abs,float(delta.max().item()))
                    max_excess=max(max_excess,float((delta-allowed).clamp_min(0).max().item()))
                    outside+=int((delta>allowed).sum().item())
    passed=same_shape and same_dtype and nonfinite==0
    passed=passed and (outside==0 if relation=='allclose' else bitwise if relation=='bitwise_equal' else not bitwise)
    row={'schema':'xiaomai.layer-comparison.v083','name':name,'run_spec_sha256':spec_sha,
         'relation':relation,'passed':bool(passed),'actual_sha256':actual_sha,
         'reference_sha256':reference_sha,'actual_shape':list(actual.shape),
         'reference_shape':list(reference.shape),'actual_dtype':str(actual.dtype),
         'reference_dtype':str(reference.dtype),'same_shape':same_shape,'same_dtype':same_dtype,
         'elements':actual.numel(),'nonfinite_pairs':nonfinite,'bitwise_equal':bitwise,
         'max_abs':max_abs if same_shape and not nonfinite and math.isfinite(max_abs) else None,
         'max_tolerance_excess':max_excess if same_shape and not nonfinite and math.isfinite(max_excess) else None,
         'elements_outside_tolerance':outside,'atol':atol,'rtol':rtol,
         'comparison_seconds':time.monotonic()-started,'phase_seconds':phase_seconds,
         'values_sampled':False,'current_true_restored':False}
    path=write_json_new(Path(output)/('COMPARISON_'+name+'.json'),row)
    if progress:progress.emit('COMPARISON_RECORDED',phase=name,receipt=path.name,passed=bool(passed))
    return row


def compare_identities(output,name,actual,reference,*,spec_sha=None,progress=None):
    """State/optimizer identities have no invented scalar numerical distance."""
    started=time.monotonic()
    row={'schema':'xiaomai.identity-comparison.v083','name':name,'run_spec_sha256':spec_sha,
         'relation':'data_identity_equal','passed':actual==reference,
         'actual_sha256':data_hash(actual),'reference_sha256':data_hash(reference),
         'max_abs':None,'numeric_distance_applicable':False,
         'comparison_seconds':time.monotonic()-started,'current_true_restored':False}
    path=write_json_new(Path(output)/('COMPARISON_'+name+'.json'),row)
    if progress:progress.emit('COMPARISON_RECORDED',phase=name,receipt=path.name,passed=row['passed'])
    return row


def pack_output(tensor):
    """Small continuation outputs, never the full parameter set."""
    import torch
    if tensor.numel()*tensor.element_size()>16<<20:
        raise ResourceLimit('COMPARISON_OUTPUT_EXCEEDS_16_MIB')
    with torch.no_grad(),torch._C._DisableTorchDispatch():
        raw=tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype).removeprefix('torch.'),
            'raw_hex':raw.hex(),'sha256':hashlib.sha256(raw).hexdigest()}


def unpack_output(row, reference):
    import torch
    if row.get('shape')!=list(reference.shape) or row.get('dtype')!=str(reference.dtype).removeprefix('torch.'):
        raise AssetError('CONTINUATION_OUTPUT_FORMAT_MISMATCH')
    size=reference.numel()*reference.element_size()
    if size>16<<20 or type(row.get('raw_hex')) is not str or len(row['raw_hex'])!=size*2:
        raise AssetError('CONTINUATION_OUTPUT_SIZE_MISMATCH')
    raw=bytes.fromhex(row['raw_hex'])
    if hashlib.sha256(raw).hexdigest()!=row.get('sha256'):
        raise AssetError('CONTINUATION_OUTPUT_HASH_MISMATCH')
    return torch.frombuffer(bytearray(raw),dtype=reference.dtype).clone().reshape(reference.shape)


def checkpoint_plan(data,output,budget,*,progress=None):
    """Measure storage, not millions of scalar objects; reserve bounded metadata.

    save_state writes raw deduplicated blobs and a small reference graph. It
    does not convert model weights into Python lists of readable numbers.
    """
    import torch
    import numpy as np
    import shutil
    import psutil
    from .limits import directory_bytes
    seen=set();storages={};auxiliary=0
    def visit(value):
        nonlocal auxiliary
        if id(value) in seen:return
        seen.add(id(value))
        if type(value) in (torch.Tensor,torch.nn.Parameter):
            storage=value.untyped_storage()
            storages[(str(value.device),storage._cdata)]=storage.nbytes()
            if value.grad is not None:visit(value.grad)
            visit(value.__dict__)
        elif type(value) is np.ndarray:auxiliary+=value.nbytes
        elif type(value) is bytes:auxiliary+=len(value)
        elif type(value) is dict:
            for key,item in value.items():visit(key);visit(item)
        elif type(value) in (tuple,list):
            for item in value:visit(item)
    visit(data)
    used=directory_bytes(budget.output);payload=sum(storages.values())+auxiliary
    reserve=1<<20
    row={'schema':'xiaomai.checkpoint-plan.v083','format':'XMO-DATA-GRAPH-2_RAW_STORAGE_BLOBS',
         'scalar_list_weight_export':False,'unique_storages':len(storages),
         'storage_payload_bytes_before_content_dedup':sum(storages.values()),
         'auxiliary_bytes':auxiliary,'largest_storage_bytes':max(storages.values(),default=0),
         'metadata_reserve_bytes':reserve,'output_bytes_before_save':used,
         'estimated_additional_disk_bytes':payload+reserve,
         'remaining_output_budget_bytes':budget.limits.max_output_bytes-used,
         'remaining_seconds':max(0,budget.limits.seconds-(time.monotonic()-budget.started)),
         'rss_sample_bytes':psutil.Process().memory_info().rss,
         'child_reload_and_parent_memory_overlap_expected':True,
         'time_to_complete_estimated':False,'trace_growth_included_in_estimate':False}
    write_json_new(Path(output)/'CHECKPOINT_PLAN.json',row)
    if progress:progress.emit('CHECKPOINT_PLAN_RECORDED',phase='save',storage_payload_bytes=payload)
    if payload+reserve>row['remaining_output_budget_bytes']:
        raise ResourceLimit('CHECKPOINT_OUTPUT_BUDGET_INSUFFICIENT')
    if shutil.disk_usage(output).free-payload-reserve<budget.limits.minimum_free_bytes:
        raise ResourceLimit('CHECKPOINT_FREE_DISK_INSUFFICIENT')
    budget.check(force=True)
    return row


def _empty_layer(text_config, dtype):
    import torch
    from tools.native_decoder_fixture import definitions
    ns, source = definitions()
    c = dict(text_config)
    if c['layer_types'][0] != 'linear_attention' or c.get('hidden_act') != 'silu':
        raise AssetError('ONLY_GDN_FIRST_LAYER_WITH_SILU_SUPPORTED')
    c['_attn_implementation'] = 'eager'
    config = types.SimpleNamespace(**c)
    with torch.device('meta'):
        layer = ns['Qwen3_5DecoderLayer'](config, 0)
    # We reuse verified class bodies, not the synthetic DecoderStack/config/cache.
    origin = {
        'source_sha256': source['source_sha256'], 'definitions': source['definitions'],
        'full_source_compiled_definition_code_retained': True,
        'parent_boundary': 'nn.Module for eager no-gradient-checkpointing class dispatch',
        'cache': 'NONE; no reference cache used',
        'position_embeddings': 'None; layer 0 is GDN and does not consume RoPE',
        'norm_policy': 'PINNED_ANCESTOR_RMS_NOT_TRUE_STABLE_WRAPPER',
        'current_true_restored': False,
    }
    return layer, origin


def build_first_layer(snapshot, *, dtype_name='float32', tick=None, progress=None):
    import torch
    if dtype_name not in ('float32','float64','bfloat16'):
        raise AssetError('UNSUPPORTED_CPU_DTYPE')
    dtype = getattr(torch, dtype_name)
    layer, origin = _empty_layer(snapshot.config['text_config'], dtype)
    expected = dict(layer.state_dict())
    names = {n[len(PREFIX):]: n for n in snapshot.tensors if n.startswith(PREFIX)}
    if set(names) != set(expected):
        raise AssetError('COMPLETE_LAYER_PARAMETER_SET_MISMATCH:' + json.dumps({
            'missing':sorted(set(expected)-set(names)), 'extra':sorted(set(names)-set(expected))}))
    state = {}; census = []
    for name, target in expected.items():
        if tick: tick()
        full = names[name]; row = snapshot.tensors[full]
        if row['shape'] != list(target.shape) or row['dtype'] != 'BF16':
            raise AssetError('REAL_LAYER_SHAPE_OR_DTYPE_MISMATCH:' + full)
        if progress:progress.emit('TENSOR_LOAD_BEGIN',phase='layer_assembly',name=full,shape=row['shape'],bytes=row['tensor_bytes'])
        raw = snapshot.raw(full)
        value = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(row['shape']).to(dtype).clone()
        state[name] = value
        census.append({'name':full, **row, 'raw_tensor_sha256':hashlib.sha256(raw).hexdigest(),
                       'loaded_dtype':str(dtype), 'loaded_value_sha256':tensor_hash(value)})
        if progress:progress.emit('TENSOR_LOAD_END',phase='layer_assembly',name=full,loaded_dtype=dtype_name)
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval(); layer.requires_grad_(False)
    if any(p.device.type != 'cpu' for p in layer.parameters()):
        raise AssetError('CPU_LAYER_LOAD_FAILED')
    snapshot.recheck()
    return layer, {**origin, 'tensors':census,
        'parameter_count':sum(p.numel() for p in layer.parameters()),
        'hidden_size':snapshot.config['text_config']['hidden_size'],
        'loaded_dtype':dtype_name, 'all_layer_weights_loaded':True,
        'asset_identity':snapshot.receipt['identity']}


def embedding_input(snapshot, ids, *, dtype_name='float32'):
    import torch
    row = snapshot.tensors[EMBEDDING]
    if row['dtype'] != 'BF16' or len(row['shape']) != 2:
        raise AssetError('EMBEDDING_FORMAT_MISMATCH')
    vocabulary, width = row['shape']
    if width != snapshot.config['text_config']['hidden_size']:
        raise AssetError('EMBEDDING_WIDTH_MISMATCH')
    if type(ids) is not list or not ids or any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
        raise AssetError('VALID_NONEMPTY_TOKEN_IDS_REQUIRED')
    values = []; origins = []
    for ident in ids:
        raw = snapshot.raw(EMBEDDING, start=ident*width, count=width)
        values.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(getattr(torch,dtype_name)).clone())
        origins.append({'token_id':ident, 'name':EMBEDDING, 'shard':row['shard'],
                        'absolute_offset':row['absolute_offset']+ident*width*2,
                        'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
    return torch.stack(values,dim=0).unsqueeze(0), origins


def select_ids(snapshot, *, ids_path=None, text=None, max_tokens=2):
    if (ids_path is None) == (text is None):
        raise AssetError('EXACTLY_ONE_INPUT_SOURCE_REQUIRED')
    if ids_path is not None:
        from .real_assets import json_data, checked_digest
        raw = Path(ids_path).read_bytes()
        ids = json_data(raw)
        origin = {'mode':'EXPLICIT_TOKEN_IDS_NOT_CLAIMED_AS_TOKENIZED_PROMPT',
                  'input_file':str(ids_path), 'sha256':hashlib.sha256(raw).hexdigest()}
    else:
        from .long_paths import asset_child
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise AssetError('TOKENIZERS_DEPENDENCY_REQUIRED_FOR_TEXT_INPUT') from e
        tokenizer = Tokenizer.from_str(snapshot.metadata('tokenizer.json').decode('utf-8'))
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        origin = {'mode':'PINNED_TOKENIZER_RAW_TEXT_NO_CHAT_TEMPLATE',
                  'text_utf8_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
                  'tokenizer_sha256':snapshot.files['tokenizer.json']['sha256']}
    if type(ids) is not list or not 1 <= len(ids) <= max_tokens or any(type(i) is not int for i in ids):
        raise AssetError('TOKEN_COUNT_OR_TYPE_NOT_ADMITTED; no silent truncation')
    return ids, origin


def forward(layer, inputs):
    return layer(inputs, position_embeddings=None, attention_mask=None, past_key_values=None)


def data_hash(value):
    """Portable identity for the explicitly admitted diagnostic state only."""
    import torch
    import numpy as np
    import struct
    def pack(v):
        if any(type(v) is t for t in (str,int,bool)) or v is None:return [type(v).__name__,v]
        if type(v) is float:return ['float64_bits',struct.pack('<d',v).hex()]
        if type(v) is bytes:return ['bytes',v.hex()]
        if type(v) is torch.Tensor:return ['tensor',str(v.dtype),list(v.shape),list(v.stride()),tensor_hash(v)]
        if type(v) is np.ndarray:return ['array',v.dtype.str,list(v.shape),hashlib.sha256(v.tobytes()).hexdigest()]
        if any(type(v) is t for t in (tuple,list)):return [type(v).__name__,[pack(x) for x in v]]
        if type(v) is dict:return ['dict',[[pack(k),pack(x)] for k,x in v.items()]]
        raise AssetError('UNSUPPORTED_DIAGNOSTIC_STATE_HASH_TYPE')
    return hashlib.sha256(json.dumps(pack(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rng_probe():
    import random
    import numpy as np
    import torch
    return {'python_u64':random.getrandbits(64),'numpy_bytes':np.random.bytes(16).hex(),
            'torch_cpu_i64':int(torch.randint(0,2**62,(1,),dtype=torch.int64).item())}


def monitor_for(layer, output, name, trace_mode, spec_sha):
    from xiaomai_observe.runtime import Monitor
    roots=[ROOT/'upstream/modeling_qwen3_5.py', Path(__file__),
           ROOT/'xiaomai_python/core.py',ROOT/'xiaomai_python/dispatch.py',
           ROOT/'xiaomai_python/streaming_mm.py',ROOT/'xiaomai_observe/state.py']
    return Monitor(layer,output/name,run_id=name,request_id=(spec_sha or 'SYNTHETIC_TEST')+':'+name,
            phase=name,python_roots=roots,values='full',operators=True,
            python_lines=trace_mode=='instructions',python_instructions=trace_mode=='instructions',
            compact_locals=trace_mode=='instructions',value_transitions=trace_mode=='instructions',
            native_calls=trace_mode=='instructions',capture_locals=True,fsync_each=False)


def backend_for(progress, monitor=None, *, trace_mode='calls'):
    from .dispatch import PythonReferenceBackend
    return PythonReferenceBackend(journal=monitor.journal if monitor else None, strict=True,
                scalar_events=trace_mode=='instructions',progress=progress,streaming_matmul=True)


def learning_step(layer, inputs, selected, opt, progress, *, monitor=None, trace_mode='calls'):
    import torch
    opt.zero_grad(set_to_none=True)
    with backend_for(progress,monitor,trace_mode=trace_mode) as backend:
        loss=forward(layer,inputs).square().mean()
        loss.backward()
        if selected.grad is None or not torch.isfinite(selected.grad).all() or not torch.any(selected.grad!=0):
            raise AssetError('NO_FINITE_NONZERO_UPDATE_GRADIENT')
        opt.step()
    return {'loss':float(loss.detach()),'backend_calls':backend.calls,'optimizer_hash':data_hash(opt.state_dict())}


def continuation(layer, data, output, *, budget, progress, trace_mode, spec_sha, trace_lifecycle=True):
    import torch
    from xiaomai_observe.state import restore_rng, rng_state
    selected=dict(layer.named_parameters())[data['update_parameter']]
    selected.requires_grad_(True)
    opt=torch.optim.SGD([selected],lr=data['lr'],momentum=data['momentum'],foreach=False)
    opt.load_state_dict(data['optimizer'])
    restore_rng(data['rng'])
    with progress.phase('rng_probe'):
        probe=rng_probe()
    with progress.phase('next_input'), (monitor_for(layer,output,'next_input_trace',trace_mode,spec_sha) if trace_lifecycle else contextlib.nullcontext()) as m:
        with torch.no_grad(), backend_for(progress,m,trace_mode=trace_mode) as b:
            y=forward(layer,data['next_input']).detach().clone()
    with progress.phase('next_update'), (monitor_for(layer,output,'next_update_trace',trace_mode,spec_sha) if trace_lifecycle else contextlib.nullcontext()) as m:
        update=learning_step(layer,data['next_input'],selected,opt,progress,monitor=m,trace_mode=trace_mode)
    budget.check(force=True)
    return {'next_output_sha256':tensor_hash(y),'next_parameters_sha256':parameter_hashes(layer),
            'next_optimizer_sha256':data_hash(opt.state_dict()),'next_rng_sha256':data_hash(rng_state()),
            'rng_probe':probe,'next_update':update,'next_backend_calls':b.calls,
            'next_output_record':pack_output(y)}


def _resume_saved(state_dir, expected, out, *, budget=None, spec_sha=None, trace_mode='calls'):
    import torch
    from xiaomai_observe.state import load_state, restore_rng
    from .progress import Progress
    output=Path(out).parent/'child_evidence'
    output.mkdir(parents=True,exist_ok=False)
    budget=budget or Budget(Limits(),output)
    with Progress(output/'PROGRESS.jsonl',spec_sha256=spec_sha,budget=budget) as progress:
        with progress.phase('load_saved_state'):
            data=load_state(state_dir,expected_sha256=expected,device='cpu')
            if data.get('run_spec_sha256')!=spec_sha:raise AssetError('RESUME_RUN_SPEC_MISMATCH')
            layer,origin=_empty_layer(data['text_config'],getattr(torch,data['dtype']))
            if origin['source_sha256']!=data['origin']['source_sha256']:
                raise AssetError('RESUME_NATIVE_SOURCE_MISMATCH')
            layer.load_state_dict(data['weights'],strict=True,assign=True)
            layer.eval();layer.requires_grad_(False)
        before=parameter_hashes(layer)
        with progress.phase('resume_forward'), (monitor_for(layer,output,'resume_forward_trace',trace_mode,spec_sha) if data['trace_lifecycle'] else contextlib.nullcontext()) as m:
            with torch.no_grad(), backend_for(progress,m,trace_mode=trace_mode) as b:
                result=forward(layer,data['input']).detach().clone()
        if data.get('comparison_contract')==COMPARISON_CONTRACT:
            saved_output=unpack_output(data['saved_output'],result)
            measured=compare_tensors(output,'restored_saved_output',result,saved_output,
                                     spec_sha=spec_sha,progress=progress)
            if not measured['passed']:raise AssetError('RESTORED_SAVED_OUTPUT_MISMATCH')
        advanced=continuation(layer,data,output,budget=budget,progress=progress,
                              trace_mode=trace_mode,spec_sha=spec_sha,trace_lifecycle=data['trace_lifecycle'])
        report={'schema':'xiaomai.layer-reload.v082','status':'LAYER_RELOAD_PASS','state_sha256':expected,
            'run_spec_sha256':spec_sha,'source_sha256':origin['source_sha256'],'worker_pid':os.getpid(),
            'parameters_sha256':before,'output_sha256':tensor_hash(result),
            'optimizer_restored':True,'rng_restored':True,'current_true_restored':False,
            'backend_calls':b.calls,'output_record':pack_output(result),
            'comparison_contract':data.get('comparison_contract'),**advanced}
        budget.check(force=True)
        write_json_new(out,report)
        budget.check(force=True)
        return report


def run_layer_checks(layer, inputs, origin, text_config, output, *, limits=None,
                     trace_mode='calls', update_parameter='input_layernorm.weight',
                     lr=1e-4, atol=2e-5, rtol=2e-5, dtype_name='float32',
                     next_inputs=None,budget=None,progress=None,spec_sha=None,trace_lifecycle=False):
    if not all(math.isfinite(v) and v>=0 for v in (atol,rtol)):
        raise ValueError('FINITE_NONNEGATIVE_TOLERANCES_REQUIRED')
    import torch
    from xiaomai_observe.state import save_state,rng_state,restore_rng
    from .progress import Progress
    output=Path(output)
    own_progress=progress is None
    if own_progress:output.mkdir(parents=True,exist_ok=False)
    budget=budget or Budget(limits or Limits(),output)
    if own_progress:progress=Progress(output/'PROGRESS.jsonl',spec_sha256=spec_sha,budget=budget)
    if trace_mode not in ('calls','instructions'):raise ValueError('TRACE_MODE_REQUIRED')
    params=dict(layer.named_parameters())
    if update_parameter not in params:raise AssetError('UNKNOWN_UPDATE_PARAMETER')
    if not 0<lr<1:raise AssetError('POSITIVE_BOUNDED_LEARNING_RATE_REQUIRED')
    if any(not torch.isfinite(p).all() for p in params.values()) or not torch.isfinite(inputs).all():
        raise AssetError('NONFINITE_INPUT_OR_PARAMETER')
    if next_inputs is None:
        # Compatibility for engineering callers only. Never admitted as a new
        # real-input continuation by the production validator.
        next_inputs=inputs
        next_is_new=False
    else:next_is_new=tensor_hash(inputs)!=tensor_hash(next_inputs)
    before=parameter_hashes(layer);identities={n:id(p) for n,p in params.items()}
    selected=params[update_parameter];old_value=selected.detach().clone()
    old_requires=selected.requires_grad;old_grad=selected.grad;entry_rng=rng_state()
    runs=[];result=None;comparisons={}
    def compare(name,actual,reference,*,relation='bitwise_equal',error_message=None,**tolerances):
        elapsed={r['phase']:r['seconds'] for r in runs}
        row=compare_tensors(output,name,actual,reference,relation=relation,
                            spec_sha=spec_sha,progress=progress,phase_seconds=elapsed,**tolerances)
        comparisons[name]=row
        if not row['passed']:raise AssetError(error_message or 'COMPARISON_FAILED:'+name)
        return row
    def identities_equal(name,actual,reference,error_message):
        row=compare_identities(output,name,actual,reference,spec_sha=spec_sha,progress=progress)
        comparisons[name]=row
        if not row['passed']:raise AssetError(error_message)
    def evaluate(name, *, python=False,traced=None,hook=None):
        if traced is None:traced=trace_lifecycle or name=='python_trace'
        start=time.monotonic();hook_handle=layer.linear_attn.register_forward_hook(hook) if hook else None
        try:
            with progress.phase(name):
                monitor=monitor_for(layer,output,name,trace_mode,spec_sha) if traced else contextlib.nullcontext()
                with monitor as m,torch.no_grad():
                    with backend_for(progress,m if traced else None,trace_mode=trace_mode) if python else contextlib.nullcontext() as b:
                        value=forward(layer,inputs).detach().clone()
                    row={'phase':name,'seconds':time.monotonic()-start,'output_sha256':tensor_hash(value),
                        'python_backend_enabled':python,'backend_calls':b.calls if python else [],
                        'trace_mode':trace_mode if traced else 'off'}
                if traced:row['trace_seal']=m.result
                if not torch.isfinite(value).all():raise AssetError('NONFINITE_LAYER_OUTPUT:'+name)
                runs.append(row)
                write_json_new(output/('PHASE_'+name+'.json'),row)
            return value
        finally:
            if hook_handle:hook_handle.remove()
    try:
        native=evaluate('native',traced=False)
        python=evaluate('python',python=True,traced=False)
        numeric=compare('native_python',native,python,relation='allclose',atol=atol,rtol=rtol,
                        error_message='NATIVE_PYTHON_NUMERICAL_MISMATCH')
        error=numeric['max_abs']
        traced=evaluate('python_trace',python=True)
        compare('trace_baseline',traced,python,error_message='TRACE_CHANGED_OUTPUT')
        identity=evaluate('identity',python=True,hook=lambda m,a,y:y)
        compare('identity_baseline',identity,python,error_message='IDENTITY_INTERVENTION_CHANGED_OUTPUT')
        altered=evaluate('zero_mixer',python=True,hook=lambda m,a,y:torch.zeros_like(y))
        compare('intervention_effect',altered,python,relation='bitwise_different',
                error_message='INTERVENTION_NO_EFFECT_IN_THIS_INPUT')
        reverted=evaluate('reverted',python=True)
        compare('intervention_rollback',reverted,python,error_message='INTERVENTION_ROLLBACK_MISMATCH')
        selected.requires_grad_(True)
        opt=torch.optim.SGD([selected],lr=lr,momentum=.9,foreach=False)
        with progress.phase('update'),(monitor_for(layer,output,'update_trace',trace_mode,spec_sha) if trace_lifecycle else contextlib.nullcontext()) as m:
            learning=learning_step(layer,inputs,selected,opt,progress,monitor=m,trace_mode=trace_mode)
        compare('parameter_update',selected,old_value,relation='bitwise_different',
                error_message='UPDATE_DID_NOT_CHANGE_PARAMETER')
        changed=parameter_hashes(layer)
        identities_equal('non_target_parameters',{n:v for n,v in changed.items() if n!=update_parameter},
                         {n:v for n,v in before.items() if n!=update_parameter},'NON_TARGET_PARAMETER_CHANGED')
        updated=evaluate('updated',python=True)
        compare('updated_output',updated,python,relation='bitwise_different',error_message='UPDATE_DID_NOT_CHANGE_OUTPUT')
        saved_rng=rng_state()
        data={'text_config':text_config,'dtype':dtype_name,
              'weights':{n:t.detach() for n,t in layer.state_dict().items()},'input':inputs.detach(),
              'next_input':next_inputs.detach(),'optimizer':opt.state_dict(),'rng':saved_rng,
              'lr':lr,'momentum':.9,'update_parameter':update_parameter,'origin':origin,'run_spec_sha256':spec_sha,
              'trace_lifecycle':trace_lifecycle,'comparison_contract':COMPARISON_CONTRACT,'saved_output':pack_output(updated)}
        storage_plan=checkpoint_plan(data,output,budget,progress=progress)
        save_started=time.monotonic()
        with progress.phase('save'),(monitor_for(layer,output,'save_trace',trace_mode,spec_sha) if trace_lifecycle else contextlib.nullcontext()):
            state=save_state(data,output/'saved_layer')
        from .limits import directory_bytes
        write_json_new(output/'CHECKPOINT_MEASUREMENT.json',{
            'schema':'xiaomai.checkpoint-measurement.v083','seconds':time.monotonic()-save_started,
            'saved_directory_bytes':directory_bytes(output/'saved_layer'),
            'total_output_bytes_after_save':directory_bytes(output),
            'storage_plan':storage_plan,'state':state,'current_true_restored':False})
        # Parent continues from exactly the state that was saved. The child
        # loads the immutable serialized version, not these later-mutated tensors.
        advanced=continuation(layer,data,output,budget=budget,progress=progress,
                              trace_mode=trace_mode,spec_sha=spec_sha,trace_lifecycle=trace_lifecycle)
        command=[sys.executable,'-B',str(ROOT/'tools/run_real_weight_layer.py'),
                 '--resume-layer',str(output/'saved_layer'),'--state-sha256',state['graph_sha256'],
                 '--out',str(output/'resumed.json'),'--execute','--trace-mode',trace_mode,
                 '--max-seconds',str(max(.001,budget.limits.seconds-(time.monotonic()-budget.started))),
                 '--max-rss-mib',str(budget.limits.max_rss_bytes>>20),
                 '--max-output-mib',str(budget.limits.max_output_bytes>>20),
                 '--minimum-free-mib',str(budget.limits.minimum_free_bytes>>20)]
        if spec_sha:command+=['--spec-sha256',spec_sha]
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
        with progress.phase('child_resume'):
            remaining=max(.001,budget.limits.seconds-(time.monotonic()-budget.started))
            with (output/'resume.stdout').open('xb') as stdout,(output/'resume.stderr').open('xb') as stderr:
                cp=subprocess.run(command,stdout=stdout,stderr=stderr,env=env,timeout=remaining)
            if cp.returncode!=0:raise AssetError('NEW_PROCESS_RESUME_FAILED:'+str(cp.returncode))
            resumed=json.loads((output/'resumed.json').read_text(encoding='utf-8'))
            identities_equal('resumed_parameters',resumed['parameters_sha256'],changed,'NEW_PROCESS_STATE_OR_OUTPUT_MISMATCH')
            compare('resumed_output',unpack_output(resumed['output_record'],updated),updated,
                    error_message='NEW_PROCESS_STATE_OR_OUTPUT_MISMATCH')
            parent_next=unpack_output(advanced['next_output_record'],next_inputs)
            compare('next_output',unpack_output(resumed['next_output_record'],parent_next),parent_next,
                    error_message='NEW_PROCESS_CONTINUATION_MISMATCH:next_output')
            for field in ('next_output_sha256','next_parameters_sha256','next_optimizer_sha256','next_rng_sha256','rng_probe'):
                identities_equal('continuation_'+field,resumed[field],advanced[field],
                                 'NEW_PROCESS_CONTINUATION_MISMATCH:'+field)
        with torch.no_grad():selected.copy_(old_value)
        selected.requires_grad_(old_requires);selected.grad=old_grad
        restore_rng(entry_rng)
        rollback=evaluate('final_rollback',python=True)
        compare('final_rollback',rollback,python,error_message='FINAL_ROLLBACK_MISMATCH')
        identities_equal('rollback_parameters',parameter_hashes(layer),before,'FINAL_ROLLBACK_MISMATCH')
        identities_equal('parameter_objects',{n:id(p) for n,p in layer.named_parameters()},identities,'PARAMETER_IDENTITY_CHANGED')
        result={'schema':'xiaomai.layer-result.v082','status':'LAYER_DIAGNOSTIC_PASS',
            'run_spec_sha256':spec_sha,'revision':REVISION,'worker_pid':os.getpid(),
            'real_assets':origin['asset_identity']==REVISION,'asset_identity':origin['asset_identity'],
            'all_layer_weights_loaded':True,'native_python_max_abs':error,'atol':atol,'rtol':rtol,
            'comparison_contract':COMPARISON_CONTRACT,'comparisons':comparisons,
            'trace_output_bitwise_equal':True,'identity_equal':True,'intervention_effect':True,'intervention_restored':True,
            'update_parameter':update_parameter,'updated_parameters':1,'updated_parameter_elements':selected.numel(),
            'optimizer':'SGD_MOMENTUM_0.9','update_effect':True,'save_reload_new_process':True,
            'parameter_identity_preserved':True,'rollback_equal':True,'runs':runs,'learning_backend_calls':learning['backend_calls'],
            'save':state,'resume_command':command,'peak_rss_sample_bytes':budget.peak_rss,'updated_parameter_hashes':changed,
            'continuation_result':{k:advanced[k] for k in ('next_output_sha256','next_optimizer_sha256','next_rng_sha256','rng_probe')},
            'next_parameter_hashes':advanced['next_parameters_sha256'],'next_input_bitwise_equal':True,'next_input_distinct':next_is_new,
            'next_update_parameters_equal':True,'optimizer_equal':True,'rng_probe_equal':True,
            'current_true_restored':False,'gpu_run':False,'trace_complete_lifecycle':False,
            'trace_lifecycle_enabled':trace_lifecycle,
            'trace_scope':'selected forwards, interventions, updates, save, child reload/continuation, rollback; admission and assembly have stage progress; opaque/native boundaries remain',
            'boundaries':['ancestor layer 0 only','no native True checkpoint','no cache/lease/external services',
                          'native storage, autograd, CPython and libm remain','unobserved opaque values retained as gaps']}
    finally:
        with torch.no_grad():selected.copy_(old_value)
        selected.requires_grad_(old_requires);selected.grad=old_grad
        restore_rng(entry_rng)
        write_json_new(output/'ROLLBACK.json',{'parameters_restored':parameter_hashes(layer)==before,
            'parameter_identity_preserved':{n:id(p) for n,p in layer.named_parameters()}==identities,
            'rng_restored':data_hash(rng_state())==data_hash(entry_rng),'original_snapshot_written':False})
        if own_progress:progress.close()
    budget.check(force=True)
    write_json_new(output/'RESULT.json',result)
    budget.check(force=True)
    return result


def run_admitted_layer(snapshot_path, output, *, ids_path=None,text=None,next_ids_path=None,next_text=None,limits=None,
                       trace_mode='calls',dtype_name='float32',update_parameter='input_layernorm.weight',
                       lr=1e-4,atol=2e-5,rtol=2e-5,spec_sha=None):
    from .real_assets import verify_snapshot
    from .progress import Progress
    from .receipt_contract import content_identity,seal_evidence
    limits=limits or Limits();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    budget=Budget(limits,output)
    snapshot=None
    with Progress(output/'PROGRESS.jsonl',spec_sha256=spec_sha,budget=budget) as progress:
        try:
            with progress.phase('source_admission'):
                snapshot=verify_snapshot(snapshot_path,tick=budget.check,progress=progress.asset)
                write_json_new(output/'ASSET_ADMISSION.json',snapshot.receipt)
            ids,input_origin=select_ids(snapshot,ids_path=ids_path,text=text,max_tokens=limits.max_tokens)
            next_ids,next_origin=select_ids(snapshot,ids_path=next_ids_path,text=next_text,max_tokens=limits.max_tokens)
            if next_ids==ids:raise AssetError('DISTINCT_EXPLICIT_NEXT_INPUT_REQUIRED')
            with progress.phase('layer_assembly'):
                layer,origin=build_first_layer(snapshot,dtype_name=dtype_name,tick=budget.check,progress=progress)
            with progress.phase('embedding'):
                inputs,embedding_origins=embedding_input(snapshot,ids,dtype_name=dtype_name)
                next_inputs,next_embeddings=embedding_input(snapshot,next_ids,dtype_name=dtype_name)
            origin.update(input=input_origin,embedding_rows=embedding_origins,next_input=next_origin,
                          next_embedding_rows=next_embeddings,custody=snapshot.receipt,run_spec_sha256=spec_sha)
            write_json_new(output/'ORIGIN.json',origin)
            result=run_layer_checks(layer,inputs,origin,snapshot.config['text_config'],output,limits=limits,
                trace_mode=trace_mode,dtype_name=dtype_name,update_parameter=update_parameter,
                lr=lr,atol=atol,rtol=rtol,next_inputs=next_inputs,budget=budget,progress=progress,spec_sha=spec_sha,trace_lifecycle=True)
            with progress.phase('source_postcheck'):
                files=snapshot.verify_all()
                write_json_new(output/'SOURCE_RECHECK.json',{'schema':'xiaomai.source-recheck.v082',
                    'status':'UNCHANGED_FULL_HASH_SET','revision':REVISION,'run_spec_sha256':spec_sha,
                    'files':files,'content_identity':content_identity(files)})
        finally:
            if snapshot:snapshot.close()
    # The model progress is already closed and becomes sealed evidence. Keep
    # finalization heartbeat events outside that tree to avoid self-mutation.
    with Progress(output.parent/'SEAL_PROGRESS.jsonl',spec_sha256=spec_sha) as sealing_progress:
        with sealing_progress.phase('seal_generated_evidence'):
            seal_evidence(output,spec_sha,tick=budget.check,progress=sealing_progress.work)
    budget.check(force=True)
    return result
