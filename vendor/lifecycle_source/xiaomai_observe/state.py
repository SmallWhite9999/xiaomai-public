"""Non-executable graph persistence for supported data, aliases, and RNG.

This is NOT a replacement for True's native object/source checkpoint loader.
Arbitrary classes, callables, sockets and credentials are intentionally refused.
"""
from __future__ import annotations
from xiaomai_observe.passive import type_is_one_of, actual_isinstance, qualified_type
import json
import math
from pathlib import Path
import random
import struct
import torch
import numpy as np
from .journal import canonical,digest,new_file
from .readable import DTYPES,TORCH_NAMES

class StateCodecError(ValueError):pass

def save_state(value,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'blobs').mkdir()
    nodes=[];memo={};storages={};storage_objs=[]
    def validate_key(k):
        # Scalar NaNs are not memoized; recreating them changes dict lookup identity.
        if type(k) is float and math.isnan(k):
            raise StateCodecError('NAN_DICTIONARY_KEY_NOT_PORTABLE')
        if type(k) is tuple:
            for part in k: validate_key(part)
    def blob(raw):
        h=digest(raw);p=out/'blobs'/h
        if not p.exists():new_file(p,raw)
        return {'sha256':h,'bytes':len(raw)}
    def encode(v):
        if v is None or type_is_one_of(type(v),(bool,int,str)):return {'scalar':v}
        if type(v) is float:return {'float_bits_le':struct.pack('<d',v).hex()}
        if type(v) is bytes:return {'bytes':blob(v)}
        if id(v) in memo:return {'ref':memo[id(v)]}
        n=len(nodes);memo[id(v)]=n;nodes.append(None)
        if actual_isinstance(v,torch.Tensor):
            if not type_is_one_of(type(v),(torch.Tensor,torch.nn.Parameter)):
                raise StateCodecError('TENSOR_SUBCLASS_NOT_ADMITTED')
            if v.grad_fn is not None:
                raise StateCodecError('LIVE_AUTOGRAD_GRAPH_NOT_PERSISTENT_DATA')
            if v.is_conj() or v.is_neg():
                raise StateCodecError('UNSUPPORTED_LAZY_TENSOR_VIEW_BITS')
            if getattr(v,'_backward_hooks',None) or getattr(v,'_post_accumulate_grad_hooks',None):
                raise StateCodecError('TENSOR_HOOKS_REQUIRE_NATIVE_CHECKPOINT')
            if v.layout!=torch.strided or v.device.type=='meta' or v.is_quantized or v.dtype not in TORCH_NAMES:
                raise StateCodecError('UNSUPPORTED_TENSOR')
            s=v.untyped_storage();key=(str(v.device),s._cdata)
            if key not in storages:
                raw=torch.empty(0,dtype=torch.uint8,device=v.device).set_(s,0,(s.nbytes(),),(1,)).cpu().numpy().tobytes()
                storages[key]=len(storage_objs)
                storage_objs.append({'data':blob(raw),'original_device':str(v.device)})
            node={'kind':'parameter' if isinstance(v,torch.nn.Parameter) else 'tensor',
                  'storage':storages[key],'shape':list(v.shape),'stride':list(v.stride()),
                  'offset':v.storage_offset(),'dtype':TORCH_NAMES[v.dtype],
                  'requires_grad':v.requires_grad,
                  'gradient':encode(v.grad),
                  'attributes':encode(dict(v.__dict__))}
        elif type(v) is np.ndarray:
            if v.dtype.hasobject or not v.flags.c_contiguous or v.base is not None:
                raise StateCodecError('UNSUPPORTED_NUMPY_OBJECT_OR_VIEW')
            node={'kind':'numpy_array','dtype':v.dtype.str,'shape':list(v.shape),'data':blob(v.tobytes())}
        elif type_is_one_of(type(v),(dict,list,tuple)):
            if type(v) is dict:
                for key in v: validate_key(key)
            node={'kind':type(v).__name__,'items':([ [encode(k),encode(x)] for k,x in v.items()]
                  if type(v) is dict else [encode(x) for x in v])}
        else:
            raise StateCodecError('UNSUPPORTED_OBJECT: '+qualified_type(type(v)))
        nodes[n]=node;return {'ref':n}
    root=encode(value)
    graph={'format':'XMO-DATA-GRAPH-2','root':root,'nodes':nodes,'storages':storage_objs,
           'native_true_object_graph':False}
    b=canonical(graph);new_file(out/'STATE.json',b)
    new_file(out/'STATE.sha256',(digest(b)+'\n').encode())
    return {'graph_sha256':digest(b),'nodes':len(nodes),'storages':len(storage_objs)}

def load_state(out,*,device='cpu',expected_sha256=None):
    out=Path(out);b=(out/'STATE.json').read_bytes();actual=digest(b)
    expected=expected_sha256 or (out/'STATE.sha256').read_text().strip()
    if actual!=expected:raise StateCodecError('STATE_HASH_MISMATCH')
    graph=json.loads(b)
    if graph.get('format') not in ('XMO-DATA-GRAPH-1','XMO-DATA-GRAPH-2'):raise StateCodecError('UNKNOWN_FORMAT')
    def data(ref):
        h=ref['sha256']
        if len(h)!=64 or any(x not in '0123456789abcdef' for x in h):raise StateCodecError('BAD_BLOB_NAME')
        p=out/'blobs'/h
        if p.is_symlink():raise StateCodecError('SYMLINK_BLOB_REJECTED')
        raw=p.read_bytes()
        if len(raw)!=ref['bytes'] or digest(raw)!=h:raise StateCodecError('STATE_BLOB_MISMATCH')
        return raw
    storage=[]
    for row in graph['storages']:
        raw=data(row['data'])
        tensor=torch.frombuffer(bytearray(raw),dtype=torch.uint8).clone() if raw else torch.empty(0,dtype=torch.uint8)
        storage.append(tensor.to(device).untyped_storage())
    memo={};active=set()
    def decode(v):
        if 'scalar' in v:return v['scalar']
        if 'float_bits_le' in v:
            try: raw=bytes.fromhex(v['float_bits_le'])
            except (ValueError,TypeError):raise StateCodecError('INVALID_FLOAT_BITS') from None
            if len(raw)!=8:raise StateCodecError('INVALID_FLOAT_BITS')
            return struct.unpack('<d',raw)[0]
        if 'float_hex' in v:return float.fromhex(v['float_hex'])
        if 'bytes' in v:return data(v['bytes'])
        i=v['ref']
        if i in memo:return memo[i]
        if i in active:raise StateCodecError('IMMUTABLE_CONTAINER_CYCLE_NOT_SUPPORTED')
        active.add(i);n=graph['nodes'][i];kind=n['kind']
        if kind=='dict':
            result={};memo[i]=result
            for k,x in n['items']:result[decode(k)]=decode(x)
        elif kind=='list':
            result=[];memo[i]=result;result.extend(decode(x) for x in n['items'])
        elif kind=='tuple':result=tuple(decode(x) for x in n['items'])
        elif kind=='numpy_array':
            dtype=np.dtype(n['dtype'])
            if dtype.hasobject:raise StateCodecError('OBJECT_ARRAY_REJECTED')
            result=np.frombuffer(data(n['data']),dtype=dtype).copy().reshape(n['shape'])
        elif kind in ('tensor','parameter'):
            dtype=DTYPES[n['dtype']][0];s=storage[n['storage']]
            shape=n['shape'];stride=n['stride'];offset=n['offset']
            if len(shape)!=len(stride) or offset<0 or any(x<0 for x in shape+stride):raise StateCodecError('INVALID_TENSOR_VIEW')
            size=DTYPES[n['dtype']][1]
            last=offset+sum((a-1)*b for a,b in zip(shape,stride)) if math.prod(shape)>0 else offset-1
            if (last+1)*size>s.nbytes():raise StateCodecError('TENSOR_VIEW_OUT_OF_BOUNDS')
            result=torch.empty(0,dtype=dtype,device=device).set_(s,offset,tuple(shape),tuple(stride))
            if kind=='parameter':result=torch.nn.Parameter(result,requires_grad=n['requires_grad'])
            elif n['requires_grad']:result.requires_grad_(True)
            # Register before decoding extras: attributes may refer to this tensor.
            memo[i]=result
            if 'attributes' in n:
                attributes=decode(n['attributes'])
                if type(attributes) is not dict:raise StateCodecError('TENSOR_ATTRIBUTES_MUST_BE_DATA_DICT')
                result.__dict__.update(attributes)
            if 'gradient' in n:
                gradient=decode(n['gradient'])
                if gradient is not None:
                    if type(gradient) is not torch.Tensor or gradient.shape!=result.shape or gradient.dtype!=result.dtype:
                        raise StateCodecError('INVALID_TENSOR_GRADIENT')
                result.grad=gradient
        else:raise StateCodecError('UNSUPPORTED_NODE_KIND')
        memo[i]=result;active.remove(i);return result
    return decode(graph['root'])

def rng_state():
    value={'python':random.getstate(),'torch_cpu':torch.random.get_rng_state(),
           'cuda_initialized':torch.cuda.is_initialized(), 'numpy_legacy':np.random.get_state()}
    if value['cuda_initialized']:value['torch_cuda']=torch.cuda.get_rng_state_all()
    # Do not initialize CUDA just to inspect RNG.
    return value

def restore_rng(value):
    random.setstate(value['python']);torch.random.set_rng_state(value['torch_cpu'].cpu())
    if 'numpy_legacy' in value:np.random.set_state(value['numpy_legacy'])
    if value.get('cuda_initialized'):
        if not torch.cuda.is_initialized():raise StateCodecError('CUDA_NOT_INITIALIZED_FOR_RNG_RESTORE')
        torch.cuda.set_rng_state_all([x.cpu() for x in value['torch_cuda']])
