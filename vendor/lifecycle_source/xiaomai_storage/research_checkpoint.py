"""Readable checkpoint data for the admitted pinned research runtime.

No class/code serialization, pickle, torch.save/load or process-local parameter
IDs enter the payload. Restore keeps the caller's model/Parameter objects and
invalidates recomputable latent caches. Observed event IDs are immutable:
restore merges saved and already disclosed events rather than erasing evidence.
This is a bounded CPU data interface, not a complete True checkpoint/lease codec.
"""
from __future__ import annotations
import hashlib,inspect,json,math,random,types
from pathlib import Path
import numpy as np
import torch
from . import kernels as k
from .adam_optimizer import PythonAdamW,OPTIONS
from .data_snapshot import clone_data,graph_signature
from .literal_state import Limits
from .code_identity import code_fingerprint

SCHEMA='XIAOMAI_PINNED_RESEARCH_DATA_V1'
ROOT=Path(__file__).resolve().parents[1]
class ResearchCheckpointError(ValueError):pass

def _emit(sink,kind,**fields):
    if sink is not None:sink(kind,**fields)

def _plain(value):
    t=type(value)
    if value is None or t in (str,bool,int):return value
    if t is float:
        if not math.isfinite(value):raise ResearchCheckpointError('NONFINITE_CONFIGURATION')
        return value
    if t in (list,tuple):return [_plain(x) for x in value]
    if t is types.SimpleNamespace:return _plain(vars(value))
    if t is dict and all(type(n) is str for n in value):return {n:_plain(v) for n,v in sorted(value.items())}
    raise ResearchCheckpointError('CONFIGURATION_TYPE_NOT_ADMITTED')

def _sources(runtime):
    from tools.research_fixture import verify_sources
    verify_sources()  # External fixed commit/hash allowlist, not just self-labelled data.
    manifest_path=ROOT/'provenance/RESEARCH_NATIVE.json'
    manifest=json.loads(manifest_path.read_bytes());pins={}
    for relative,row in manifest['files'].items():
        path=ROOT/'upstream/research_native'/relative
        raw=path.read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ResearchCheckpointError('PINNED_RESEARCH_SOURCE_CHANGED:'+relative)
        pins['upstream/research_native/'+relative]=row['sha256']
    native_path=ROOT/'upstream/research_native/experiments/ornith_problem_formation_v1/research_neural.py'
    if Path(inspect.getfile(type(runtime))).resolve()!=native_path.resolve():
        raise ResearchCheckpointError('RUNTIME_NOT_PINNED_SOURCE')
    compiled=compile(native_path.read_bytes(),str(native_path),'exec',dont_inherit=True)
    cls=next(c for c in compiled.co_consts if type(c) is types.CodeType and c.co_name=='ResearchNeuralRuntime')
    expected={c.co_name:code_fingerprint(c) for c in cls.co_consts if type(c) is types.CodeType}
    for name,fingerprint in expected.items():
        method=type(runtime).__dict__.get(name)
        if type(method) is property:method=method.fget
        method=inspect.unwrap(method)
        if type(method) is not types.FunctionType or code_fingerprint(method.__code__)!=fingerprint or name in vars(runtime):
            raise ResearchCheckpointError('LOADED_RESEARCH_METHOD_CHANGED:'+name)
    for relative in ('xiaomai_storage/research_checkpoint.py','xiaomai_storage/adam_optimizer.py',
                     'xiaomai_storage/data_snapshot.py','xiaomai_storage/literal_state.py','xiaomai_storage/kernels.py',
                     'tools/research_fixture.py'):
        path=ROOT/relative;pins[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'base_commit':manifest['base_commit'],'files':pins}

def _model_parts(model):
    return dict(model.named_parameters(remove_duplicate=False)),dict(model.named_buffers(remove_duplicate=False))

def _topology(parameters,buffers):
    objects={};stores={};result={}
    for kind,items in (('parameters',parameters),('buffers',buffers)):
        result[kind]={}
        for name,t in items.items():
            k.admit(t)
            if t.is_inference() or not t.is_leaf or t._backward_hooks or t._post_accumulate_grad_hooks:
                raise ResearchCheckpointError('LIVE_GRAPH_HOOK_OR_INFERENCE_TENSOR_NOT_ADMITTED:'+name)
            oid=objects.setdefault(id(t),len(objects));sid=stores.setdefault(t.untyped_storage()._cdata,len(stores))
            result[kind][name]={'shape':list(t.shape),'stride':list(t.stride()),'offset':t.storage_offset(),
                'storage_bytes':t.untyped_storage().nbytes(),'dtype':str(t.dtype),'object_group':oid,'storage_group':sid}
    return result

def _check_live(model,runtime,optimizer,*,allow_failed=False):
    if runtime.model is not model or getattr(model,'_ornith_research_neural_runtime',None) is not runtime:
        raise ResearchCheckpointError('MODEL_RUNTIME_BINDING_CHANGED')
    if runtime.binding.stack or runtime.binding.encode_depth:
        raise ResearchCheckpointError('ACTIVE_RESEARCH_CONTEXT_NOT_SAVEABLE')
    if getattr(model,'_ornith_research_cache_lease',None) is not None:
        raise ResearchCheckpointError('ACTIVE_DECODE_LEASE_NOT_SAVEABLE')
    if '_python_research_adapter' in vars(runtime):
        raise ResearchCheckpointError('ATTACHED_RESEARCH_ADAPTER_NOT_SAVEABLE')
    runtime._parameter_signature()
    if type(optimizer) is not PythonAdamW:raise ResearchCheckpointError('EXACT_PYTHON_ADAMW_REQUIRED')
    if optimizer.failed and not allow_failed:raise ResearchCheckpointError('FAILED_OPTIMIZER_NOT_SAVEABLE')
    optimizer._validate()
    parameters,buffers=_model_parts(model)
    if any(name not in parameters or parameters[name] is not p for name,p in optimizer.named.items()):
        raise ResearchCheckpointError('OPTIMIZER_MODEL_PARAMETER_BINDING_CHANGED')
    topology=_topology(parameters,buffers)
    grad_stores=set();model_stores={t.untyped_storage()._cdata for t in [*parameters.values(),*buffers.values()]}
    for name,p in parameters.items():
        if p.grad is None:continue
        g=k.admit(p.grad)
        if g.grad_fn is not None or g.requires_grad or tuple(g.shape)!=tuple(p.shape) or g.dtype is not p.dtype:
            raise ResearchCheckpointError('GRADIENT_DOMAIN_NOT_ADMITTED:'+name)
        sid=g.untyped_storage()._cdata
        if sid in model_stores or sid in grad_stores:raise ResearchCheckpointError('GRADIENT_ALIAS_NOT_ADMITTED')
        grad_stores.add(sid)
    return parameters,buffers,topology

def _configuration(model,runtime):
    return {'model_type':type(model).__module__+'.'+type(model).__qualname__,
            'model_config':_plain(vars(model.config)), 'layer':runtime.layer,
            'width':runtime.components.core.width,'slots':runtime.components.core.slots,'steps':runtime.components.steps,
            'module_types':{n:type(m).__module__+'.'+type(m).__qualname__ for n,m in model.named_modules()},
            'loaded_forwards':_forward_identities(model)}

def _forward_identities(model):
    """Portable identity of actual module forward code and its admitted binding.

    Include wrappers, defaults and admitted closure cells, rather than trusting
    an unchanged class name or unwrapping away executable decorator code. This
    is not a serialization of arbitrary callable objects or all Python globals.
    Package and torch source filenames are normalized so a new extraction path
    is not itself a code change. Source bytes and executable code both bind.
    """
    source_cache={};function_cache={};active=set()
    torch_root=Path(torch.__file__).resolve().parent
    def source(filename):
        path=Path(filename).resolve()
        if path in source_cache:return source_cache[path]
        for base,prefix in ((ROOT,'package:'),(torch_root,'dependency:torch/')):
            try:relative=path.relative_to(base)
            except ValueError:continue
            if not path.is_file():raise ResearchCheckpointError('FORWARD_SOURCE_MISSING')
            result=(prefix+relative.as_posix(),hashlib.sha256(path.read_bytes()).hexdigest())
            source_cache[path]=result;return result
        raise ResearchCheckpointError('FORWARD_SOURCE_OUTSIDE_ADMITTED_TREE')
    def portable_code(code):
        filename,_=source(code.co_filename)
        return code.replace(co_filename=filename,co_consts=tuple(
            portable_code(c) if type(c) is types.CodeType else c for c in code.co_consts))
    def value(v):
        t=type(v)
        if v is None or t in (str,bool,int):return {'kind':t.__name__,'value':v}
        if t is float:return {'kind':'float','hex':v.hex()}
        if t in (tuple,list):return {'kind':t.__name__,'items':[value(x) for x in v]}
        if t is dict and all(type(n) is str for n in v):
            return {'kind':'dict','items':{n:value(x) for n,x in sorted(v.items())}}
        if t is types.FunctionType:return {'kind':'function','value':function(v)}
        # torch.no_grad's decorator closes over its clone bound method. It
        # creates a fresh no_grad context per call; persist its actual code and
        # exact admitted owner type/state, not a process-local owner address.
        if t is types.MethodType and type(v.__self__) is torch.no_grad:
            return {'kind':'torch.no_grad_method','method':function(v.__func__),
                    'owner_state':value(vars(v.__self__))}
        raise ResearchCheckpointError('FORWARD_CLOSURE_OR_DEFAULT_NOT_ADMITTED')
    def function(fn):
        if type(fn) is not types.FunctionType:raise ResearchCheckpointError('PYTHON_FORWARD_FUNCTION_REQUIRED')
        if id(fn) in function_cache:return function_cache[id(fn)]
        if id(fn) in active:raise ResearchCheckpointError('CYCLIC_FORWARD_CALLABLE_NOT_ADMITTED')
        active.add(id(fn))
        try:
            origin,digest=source(fn.__code__.co_filename)
            result={'module':fn.__module__,'qualname':fn.__qualname__,'source':origin,'source_sha256':digest,
                    'code_sha256':code_fingerprint(portable_code(fn.__code__)),
                    'defaults':value(fn.__defaults__),'kwdefaults':value(fn.__kwdefaults__),
                    'closure':{n:value(c.cell_contents) for n,c in zip(fn.__code__.co_freevars,fn.__closure__ or ())}}
            if '__wrapped__' in vars(fn):result['wrapped']=value(vars(fn)['__wrapped__'])
            function_cache[id(fn)]=result;return result
        finally:active.remove(id(fn))
    result={}
    for name,module in model.named_modules():
        owner=next((cls for cls in type(module).__mro__ if 'forward' in vars(cls)),None)
        if owner is None:raise ResearchCheckpointError('FORWARD_CLASS_BINDING_MISSING:'+name)
        class_forward=vars(owner)['forward']
        if type(class_forward) is not types.FunctionType:raise ResearchCheckpointError('FORWARD_DESCRIPTOR_NOT_ADMITTED:'+name)
        binding='class_bound'
        if 'forward' in vars(module):
            bound=vars(module)['forward'];binding='instance_bound'
            if type(bound) is not types.MethodType or bound.__self__ is not module:
                raise ResearchCheckpointError('FORWARD_INSTANCE_BINDING_NOT_ADMITTED:'+name)
            fn=bound.__func__
        else:fn=class_forward
        if getattr(module,'_compiled_call_impl',None) is not None:
            raise ResearchCheckpointError('COMPILED_MODULE_CALL_NOT_ADMITTED:'+name)
        result[name]={'binding':binding,'owner_type':owner.__module__+'.'+owner.__qualname__,
                      'class_forward':function(class_forward),'actual_forward':function(fn)}
    return result

def _rng():
    n=np.random.get_state()
    return {'python':random.getstate(),'torch':torch.get_rng_state(),
            'numpy':(n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4]))}

def _validate_rng(data):
    if type(data) is not dict or set(data)!={'python','torch','numpy'}:raise ResearchCheckpointError('RNG_SCHEMA')
    random.Random().setstate(data['python'])
    torch.Generator(device='cpu').set_state(data['torch'])
    n=data['numpy']
    if type(n) is not tuple or len(n)!=5 or type(n[1]) is not list:raise ResearchCheckpointError('NUMPY_RNG_SCHEMA')
    if any(type(x) is not int or not 0<=x<=0xffffffff for x in n[1]):raise ResearchCheckpointError('NUMPY_RNG_WORDS')
    np.random.RandomState().set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))

def _set_rng(data):
    random.setstate(data['python']);torch.set_rng_state(data['torch']);n=data['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))

def _ledger(runtime,snapshot,ledger):
    if type(snapshot) is not dict or set(snapshot)!={'schema','token_rows','event_ids','concept_version','source_version'}:
        raise ResearchCheckpointError('SNAPSHOT_SCHEMA')
    validated=runtime._validate_snapshot(snapshot)
    if type(ledger) is not dict or any(type(n) is not str or not n for n in ledger):
        raise ResearchCheckpointError('EVENT_LEDGER_SCHEMA')
    check={'schema':'research-neural-input-v1','token_rows':list(ledger.values()),'event_ids':list(ledger),
           'concept_version':validated['concept_version'],'source_version':validated['source_version']}
    runtime._validate_snapshot(check)
    for name,tokens in zip(validated['event_ids'],validated['token_rows']):
        if name not in ledger or ledger[name]!=tokens:raise ResearchCheckpointError('SNAPSHOT_EVENT_LEDGER_MISMATCH')
    return validated

def _payload(model,runtime,optimizer,*,allow_failed=False,external_base=None):
    params,buffers,topology=_check_live(model,runtime,optimizer,allow_failed=allow_failed)
    external=None
    if external_base is not None:
        from .external_research_base import FrozenExternalBase
        if type(external_base) is not FrozenExternalBase:raise ResearchCheckpointError('EXACT_EXTERNAL_BASE_BINDING_REQUIRED')
        external=external_base.verify(model,runtime,optimizer)
        params={n:p for n,p in params.items() if n in external_base.research_parameters}
        buffers={n:b for n,b in buffers.items() if n in external_base.research_buffers}
        topology=_topology(params,buffers)
    _ledger(runtime,runtime._snapshot,runtime._event_tokens)
    if type(runtime.weight_revision) is not int or runtime.weight_revision<0 or type(runtime.state_recomputations) is not int or runtime.state_recomputations<0:
        raise ResearchCheckpointError('RUNTIME_COUNTER_SCHEMA')
    detached={}
    def tensor(t):
        if id(t) not in detached:detached[id(t)]=t.detach()
        return detached[id(t)]
    result={'schema':SCHEMA,'sources':_sources(runtime),'configuration':_configuration(model,runtime),
        'topology':topology,'parameters':{n:tensor(t) for n,t in params.items()},'buffers':{n:tensor(t) for n,t in buffers.items()},
        'gradients':{n:None if p.grad is None else tensor(p.grad) for n,p in params.items()},
        'requires_grad':{n:p.requires_grad for n,p in params.items()},
        'module_training':{n:m.training for n,m in model.named_modules()},'optimizer':optimizer.state_dict(),
        'runtime':{'snapshot':runtime._snapshot,'event_tokens':runtime._event_tokens,'weight_revision':runtime.weight_revision,
                   'state_recomputations':runtime.state_recomputations},'rng':_rng(),
        'ephemeral':{'active_contexts':0,'encode_depth':0,'decode_lease':False,'latent_state_saved':False,'native_versions_rewound':False}}
    if external is not None:
        result['schema']='XIAOMAI_PINNED_RESEARCH_EXTERNAL_BASE_V1'
        result['external_base']=external
    return result

def content_signature(value,*,limits=Limits()):
    """Portable graph + complete storage byte identity, without process IDs."""
    # Dict construction from frozenset option names varies with PYTHONHASHSEED.
    # Normalize schema dictionary order while retaining repeated data references.
    memo={}
    def canonical(v):
        if type(v) not in (dict,list,tuple):return v
        if id(v) in memo:return memo[id(v)]
        if type(v) is dict:
            if any(type(n) is not str for n in v):raise ResearchCheckpointError('STRING_SCHEMA_KEYS_REQUIRED')
            out={};memo[id(v)]=out
            out.update((n,canonical(v[n])) for n in sorted(v));return out
        if type(v) is list:
            out=[];memo[id(v)]=out;out.extend(canonical(x) for x in v);return out
        out=tuple(canonical(x) for x in v);memo[id(v)]=out;return out
    _,_,_,ident=graph_signature(canonical(value),limits)
    return ident[0],ident[-1]

def _bindings(model,runtime,optimizer):
    p,b,t=_check_live(model,runtime,optimizer,allow_failed=True)
    return (id(runtime),id(runtime.binding),tuple((n,id(v),v.untyped_storage()._cdata) for n,v in p.items()),
            tuple((n,id(v),v.untyped_storage()._cdata) for n,v in b.items()))

def capture(model,runtime,optimizer,*,sink=None,limits=Limits(),external_base=None):
    bindings=_bindings(model,runtime,optimizer)
    original=_payload(model,runtime,optimizer,external_base=external_base);before=content_signature(original,limits=limits)
    result=clone_data(original,sink=sink,limits=limits)
    _emit(sink,'RESEARCH_CHECKPOINT_CAPTURED',parameters=len(result['parameters']),events=len(result['runtime']['event_tokens']),
          full_historical_ledger=True,latent_state_saved=False)
    if _bindings(model,runtime,optimizer)!=bindings or content_signature(_payload(model,runtime,optimizer,external_base=external_base),limits=limits)!=before:
        raise ResearchCheckpointError('LIVE_RESEARCH_STATE_CHANGED_DURING_CAPTURE')
    return result

def _prepare(model,runtime,optimizer,data,limits,external_base=None):
    live=_payload(model,runtime,optimizer,allow_failed=True,external_base=external_base)
    if type(data) is not dict or set(data)!=set(live) or data['schema']!=live['schema']:
        raise ResearchCheckpointError('CHECKPOINT_SCHEMA')
    for key in ('sources','configuration','topology','ephemeral'):
        if data[key]!=live[key]:raise ResearchCheckpointError('CHECKPOINT_'+key.upper()+'_MISMATCH')
    if external_base is not None and data['external_base']!=live['external_base']:
        raise ResearchCheckpointError('CHECKPOINT_EXTERNAL_BASE_MISMATCH')
    for key in ('parameters','buffers','gradients','requires_grad','module_training'):
        if type(data[key]) is not dict or set(data[key])!=set(live[key]):raise ResearchCheckpointError('CHECKPOINT_NAMED_'+key.upper())
    for key in ('requires_grad','module_training'):
        if any(type(v) is not bool for v in data[key].values()):raise ResearchCheckpointError('BOOL_RUNTIME_FLAG_REQUIRED')
    # Verify actual payload storage geometry and aliases, not only its declared topology.
    actual=_topology(data['parameters'],data['buffers'])
    for kind in ('parameters','buffers'):
        for name,row in actual[kind].items():
            expected=data['topology'][kind][name]
            if row!=expected:raise ResearchCheckpointError('SAVED_TENSOR_TOPOLOGY_MISMATCH:'+name)
    grad_stores=set();model_stores={t.untyped_storage()._cdata for t in [*data['parameters'].values(),*data['buffers'].values()]}
    for n,g in data['gradients'].items():
        if g is not None:
            p=data['parameters'][n];k.admit(g)
            if type(g) is not torch.Tensor or g.requires_grad or g.grad_fn is not None or tuple(g.shape)!=tuple(p.shape) or g.dtype is not p.dtype:
                raise ResearchCheckpointError('SAVED_GRADIENT_DOMAIN:'+n)
            sid=g.untyped_storage()._cdata
            if sid in model_stores or sid in grad_stores:raise ResearchCheckpointError('SAVED_GRADIENT_ALIAS_NOT_ADMITTED')
            grad_stores.add(sid)
    rd=data['runtime']
    if type(rd) is not dict or set(rd)!={'snapshot','event_tokens','weight_revision','state_recomputations'}:raise ResearchCheckpointError('RUNTIME_SCHEMA')
    _ledger(runtime,rd['snapshot'],rd['event_tokens'])
    if any(type(rd[k]) is not int or rd[k]<0 for k in ('weight_revision','state_recomputations')):raise ResearchCheckpointError('RUNTIME_COUNTER_SCHEMA')
    merged=dict(runtime._event_tokens)
    for name,tokens in rd['event_tokens'].items():
        if name in merged and merged[name]!=tokens:raise ResearchCheckpointError('IMMUTABLE_EVENT_CONTENT_CONFLICT:'+name)
        merged[name]=tokens
    _validate_rng(data['rng'])
    # Staging optimizer has its own options/moments and never updates parameters.
    groups=[{**{n:g[n] for n in OPTIONS},'group_name':g['group_name'],
             'params':[(n,optimizer.named[n]) for n in g['param_names']]} for g in optimizer.param_groups]
    probe=PythonAdamW(groups,record_values=False);probe.load_state_dict(data['optimizer'])
    staged=clone_data(data,limits=limits);staged['runtime']['event_tokens']=clone_data(merged,limits=limits)
    return staged,live,len(set(merged)-set(rd['event_tokens']))

def _apply(model,runtime,optimizer,data,external_base=None):
    params,buffers=_model_parts(model);copied=set();changed=[]
    if external_base is not None:
        params={n:p for n,p in params.items() if n in external_base.research_parameters}
        buffers={n:b for n,b in buffers.items() if n in external_base.research_buffers}
    with torch.no_grad():
        for kind,targets in (('parameters',params),('buffers',buffers)):
            for name,target in targets.items():
                store=target.untyped_storage()._cdata
                if store not in copied:
                    src=k.raw(data[kind][name]);dst=k.raw(target)
                    if len(src)!=len(dst):raise ResearchCheckpointError('RESTORE_STORAGE_SIZE_CHANGED')
                    for at in range(0,len(src),65536):dst[at:at+65536]=src[at:at+65536]
                    copied.add(store)
                changed.append(target)
        torch.autograd.graph.increment_version(changed)
        for name,p in params.items():
            p.requires_grad_(data['requires_grad'][name]);p.grad=data['gradients'][name]
        for name,module in model.named_modules():module.training=data['module_training'][name]
    optimizer.load_state_dict(data['optimizer'])
    rd=data['runtime'];runtime._snapshot=rd['snapshot'];runtime._event_tokens=rd['event_tokens']
    runtime.weight_revision=rd['weight_revision'];runtime.state_recomputations=rd['state_recomputations']
    runtime._inference_state=None;runtime._inference_key=None
    _set_rng(data['rng'])

def restore(model,runtime,optimizer,data,*,sink=None,limits=Limits(),external_base=None):
    """Restore admitted data in place; retain newly observed immutable events.

    No concurrent-writer exclusion is claimed. Synchronous observer drift is
    detected and rollback attempted; a failed rollback never returns success.
    """
    bindings=_bindings(model,runtime,optimizer)
    staged,live,retained=_prepare(model,runtime,optimizer,data,limits,external_base)
    backup=clone_data(live,limits=limits);before_failed=optimizer.failed
    expected=content_signature(staged,limits=limits)
    try:
        _emit(sink,'RESEARCH_CHECKPOINT_RESTORE_PREPARED',events=len(staged['runtime']['event_tokens']),retained_live_events=retained)
        if _bindings(model,runtime,optimizer)!=bindings or content_signature(_payload(model,runtime,optimizer,allow_failed=True,external_base=external_base),limits=limits)!=content_signature(backup,limits=limits):
            raise ResearchCheckpointError('LIVE_STATE_CHANGED_BEFORE_RESTORE')
        _apply(model,runtime,optimizer,staged,external_base)
        if _bindings(model,runtime,optimizer)!=bindings or content_signature(_payload(model,runtime,optimizer,external_base=external_base),limits=limits)!=expected:
            raise ResearchCheckpointError('RESEARCH_RESTORE_READBACK_MISMATCH')
        receipt={'status':'RESEARCH_DATA_RESTORED','parameter_objects_preserved':True,'historical_events':len(runtime._event_tokens),
                 'retained_live_events':retained,'inference_cache_cleared':True,'decode_lease_restored':False,'native_versions_rewound':False}
        if external_base is not None:
            receipt.update(external_frozen_base_verified=True,external_base_copied=False,
                           research_parameters_saved=len(staged['parameters']))
        _emit(sink,'RESEARCH_CHECKPOINT_RESTORED',**receipt)
        if _bindings(model,runtime,optimizer)!=bindings or content_signature(_payload(model,runtime,optimizer,external_base=external_base),limits=limits)!=expected:
            raise ResearchCheckpointError('OBSERVER_CHANGED_RESTORED_STATE')
        return receipt
    except BaseException as original:
        try:
            if _bindings(model,runtime,optimizer)!=bindings:raise ResearchCheckpointError('BINDING_CHANGED_NO_UNSAFE_ROLLBACK')
            _apply(model,runtime,optimizer,backup,external_base);optimizer.failed=before_failed
            if content_signature(_payload(model,runtime,optimizer,allow_failed=True,external_base=external_base),limits=limits)!=content_signature(backup,limits=limits):
                raise ResearchCheckpointError('ROLLBACK_READBACK_MISMATCH')
        except BaseException as rollback:
            optimizer.failed=True
            raise ResearchCheckpointError('RESEARCH_RESTORE_FAILED_ROLLBACK_INCOMPLETE:'+type(rollback).__name__) from original
        raise
