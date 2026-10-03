"""Non-executing Python-literal persistence of supported data/tensor storage.

No pickle, torch.save or torch.load. STATE.py and every block.py are restricted
literal assignments; loaders parse, validate and hash them, never import/exec.
Shared data storage and repeated object references are retained. The original
process's live autograd graph, version counter values/groups, classes, functions,
services and sockets are NOT reconstructed. Runtime restore uses the existing
model/optimizer objects via their admitted data interfaces.
"""
from __future__ import annotations
import ast
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from dataclasses import dataclass
import torch
from . import kernels as k, numeric_words as words

class StateError(ValueError):pass

@dataclass(frozen=True)
class Limits:
    max_storage_bytes:int=2*1024**3
    max_graph_bytes:int=64*1024**2
    max_nodes:int=200000
    block_bytes:int=16384
    max_depth:int=256
    def __post_init__(self):
        for n in (self.max_storage_bytes,self.max_graph_bytes,self.max_nodes,self.block_bytes,self.max_depth):
            if type(n) is not int or n<=0:raise StateError('POSITIVE_CONCRETE_LIMIT_REQUIRED')
        if self.block_bytes>65536:raise StateError('BLOCK_TOO_LARGE')

DTYPES={str(dt):dt for dt in k.DTYPES}
FORMAT='XIAOMAI_PYTHON_LITERAL_STATE_V1'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def emit(sink,kind,**fields):
    if sink is not None:sink(kind,**fields)

def _literal_bytes(name,value):return (name+' = '+repr(value)+'\n').encode('utf-8')
def _parse(raw,name,limit):
    if len(raw)>limit:raise StateError('LITERAL_LIMIT')
    try:
        module=ast.parse(raw,mode='exec')
        if len(module.body)!=1 or type(module.body[0]) is not ast.Assign:raise StateError('ONE_LITERAL_ASSIGNMENT_REQUIRED')
        assign=module.body[0]
        if len(assign.targets)!=1 or type(assign.targets[0]) is not ast.Name or assign.targets[0].id!=name:
            raise StateError('LITERAL_TARGET_MISMATCH')
        for node in ast.walk(assign.value):
            if type(node) is ast.Dict:
                keys=[ast.literal_eval(key) for key in node.keys]
                # Schema dictionaries use string keys. Reject duplicate keys;
                # user dictionaries are encoded as lists of typed key nodes.
                if any(type(key) is not str for key in keys) or len(keys)!=len(set(keys)):
                    raise StateError('DUPLICATE_OR_NONSTRING_SCHEMA_KEY')
        return ast.literal_eval(assign.value)
    except (SyntaxError,ValueError,TypeError,MemoryError,RecursionError) as exc:
        if type(exc) is StateError:raise
        raise StateError('NON_LITERAL_OR_MALFORMED_DATA') from exc

def _new_file(path,raw):
    with path.open('xb') as stream:
        n=stream.write(raw)
        if n!=len(raw):raise StateError('SHORT_FILE_WRITE')
        stream.flush();os.fsync(stream.fileno())

def _key_ok(v):
    t=type(v)
    if v is None or any(t is x for x in (str,bool,int,bytes)):return
    if t is float and not math.isnan(v):return
    if t is tuple:
        for x in v:_key_ok(x)
        return
    raise StateError('UNSUPPORTED_OR_NAN_DICTIONARY_KEY')

def _build_graph(value,limits):
    nodes=[];memo={};stores=[];store_ids={};owners=[];active_immutable=set()
    def encode(v,depth=0):
        if depth>limits.max_depth:raise StateError('GRAPH_DEPTH_LIMIT')
        t=type(v)
        if v is None:return {'kind':'none'}
        if t is bool:return {'kind':'bool','value':v}
        if t is int:return {'kind':'int','value':str(v)}
        if t is str:return {'kind':'str','value':v}
        if t is float:return {'kind':'float64','bits_le':struct.pack('<d',v).hex()}
        if id(v) in memo:
            if id(v) in active_immutable:raise StateError('IMMUTABLE_CYCLE_UNSUPPORTED')
            return {'ref':memo[id(v)]}
        if len(nodes)>=limits.max_nodes:raise StateError('NODE_LIMIT')
        ident=len(nodes);memo[id(v)]=ident;nodes.append(None);owners.append(v)
        enc=lambda x:encode(x,depth+1)
        if t is bytes:
            node={'kind':'bytes','hex':v.hex()}
        elif t is list or t is tuple:
            if t is tuple:active_immutable.add(id(v))
            node={'kind':'list' if t is list else 'tuple','items':[enc(x) for x in v]}
            active_immutable.discard(id(v))
        elif t is dict or t is OrderedDict:
            for key in v:_key_ok(key)
            node={'kind':'ordered' if t is OrderedDict else 'dict','items':[[enc(a),enc(b)] for a,b in v.items()]}
            if t is OrderedDict:node['attributes']=enc(v.__dict__)
        elif t is torch.Tensor or t is torch.nn.Parameter:
            k.admit(v)
            if v.grad_fn is not None:raise StateError('LIVE_AUTOGRAD_GRAPH_NOT_SERIALIZED')
            if v.is_inference():raise StateError('INFERENCE_TENSOR_RUNTIME_SEMANTICS_NOT_ADMITTED')
            if v._backward_hooks or v._post_accumulate_grad_hooks:raise StateError('TENSOR_HOOKS_REQUIRE_NATIVE_RESTORE')
            storage=v.untyped_storage();sid=storage._cdata
            if sid not in store_ids:
                count=storage.nbytes()
                if count+sum(s['nbytes'] for s in stores)>limits.max_storage_bytes:raise StateError('STORAGE_LIMIT')
                store_ids[sid]=len(stores);stores.append({'nbytes':count,'source':v,'source_storage':storage})
            node={'kind':'parameter' if t is torch.nn.Parameter else 'tensor','storage':store_ids[sid],
                  'shape':list(v.shape),'stride':list(v.stride()),'offset':v.storage_offset(),
                  'dtype':str(v.dtype),'requires_grad':v.requires_grad,
                  'gradient':enc(v.grad),'attributes':enc(v.__dict__)}
        else:raise StateError('UNSUPPORTED_DATA_OBJECT_TYPE')
        nodes[ident]=node;return {'ref':ident}
    root=encode(value)
    graph={'format':FORMAT,'root':root,'nodes':nodes,'storages':[{'nbytes':s['nbytes']} for s in stores],
           'native_true_object_graph':False,'live_autograd_restored':False,
           'version_counter_topology_restored':False,'shared_storage_restored':True}
    return graph,stores,owners

def _source_hash(store,block_bytes):
    raw=k.raw(store['source']);h=hashlib.sha256()
    if len(raw)!=store['nbytes']:raise StateError('SOURCE_STORAGE_RESIZED')
    for start in range(0,len(raw),block_bytes):h.update(raw[start:start+block_bytes])
    return h.hexdigest()

def save(value,path,*,limits=Limits(),sink=None,before_commit=None):
    """Fresh destination, check source again, publish COMMIT.json only on success."""
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False);(path/'blocks').mkdir()
    files={};graph,stores,owners=_build_graph(value,limits)
    graph_identity=digest(_literal_bytes('STATE',graph))
    if len(_literal_bytes('STATE',graph))>limits.max_graph_bytes:raise StateError('GRAPH_SIZE_LIMIT')
    versions={id(s['source']):s['source']._version for s in stores}
    try:
        for sid,store in enumerate(stores):
            initial_hash=_source_hash(store,limits.block_bytes);raw=k.raw(store['source']);chunks=[]
            for start in range(0,len(raw),limits.block_bytes):
                data=bytes(raw[start:start+limits.block_bytes]);name=f'blocks/s{sid:06d}_{start:016d}.py'
                payload={'storage':sid,'offset':start,'data':tuple(data)}
                encoded=_literal_bytes('BLOCK',payload)
                emit(sink,'STATE_BLOCK_WRITE_BEGIN',path=name,source_bytes=len(data),source_sha256=digest(data))
                _new_file(path/name,encoded);files[name]={'bytes':len(encoded),'sha256':digest(encoded)}
                chunks.append({'path':name,'offset':start,'bytes':len(data),'data_sha256':digest(data)})
                emit(sink,'STATE_BLOCK_WRITE_END',path=name,bytes=len(encoded),sha256=digest(encoded))
            graph['storages'][sid].update(chunks=chunks,data_sha256=initial_hash)
            if _source_hash(store,limits.block_bytes)!=initial_hash:raise StateError('SOURCE_CHANGED_DURING_COPY')
        rawgraph=_literal_bytes('STATE',graph);_new_file(path/'STATE.py',rawgraph)
        files['STATE.py']={'bytes':len(rawgraph),'sha256':digest(rawgraph)}
        manifest={'format':FORMAT,'files':files,'graph_sha256':digest(rawgraph),
                  'total_storage_bytes':sum(s['nbytes'] for s in stores)}
        m=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode();_new_file(path/'MANIFEST.json',m)
        if before_commit is not None:before_commit()  # fault-injection seam, never serialized
        again,again_stores,again_owners=_build_graph(value,limits)
        if [id(x) for x in owners]!=[id(x) for x in again_owners] or [s['source_storage']._cdata for s in stores]!=[s['source_storage']._cdata for s in again_stores]:raise StateError('SOURCE_OBJECT_GRAPH_CHANGED')
        if digest(_literal_bytes('STATE',again))!=graph_identity:raise StateError('SOURCE_GRAPH_CHANGED')
        for sid,store in enumerate(stores):
            if store['source']._version!=versions[id(store['source'])] or _source_hash(store,limits.block_bytes)!=graph['storages'][sid]['data_sha256']:
                raise StateError('SOURCE_CHANGED_BEFORE_COMMIT')
        # Validate every published byte before the final atomic completion marker.
        verify(path,expected_manifest_sha256=digest(m),limits=limits,require_commit=False)
        commit={'format':FORMAT,'status':'COMPLETE','manifest_sha256':digest(m)}
        _new_file(path/'COMMIT.json',json.dumps(commit,sort_keys=True).encode())
        emit(sink,'STATE_COMMITTED',manifest_sha256=digest(m),storages=len(stores),nodes=len(graph['nodes']))
        # The observer is synchronous and may fail or alter the destination.
        # Its return is part of admission, not a guarantee of observer purity.
        verify(path,expected_manifest_sha256=digest(m),limits=limits)
        final_graph,final_stores,final_owners=_build_graph(value,limits)
        if ([id(x) for x in owners]!=[id(x) for x in final_owners]
            or [s['source_storage']._cdata for s in stores]!=[s['source_storage']._cdata for s in final_stores]
            or digest(_literal_bytes('STATE',final_graph))!=graph_identity):
            raise StateError('SOURCE_GRAPH_CHANGED_IN_COMMIT_OBSERVER')
        for sid,store in enumerate(stores):
            if store['source']._version!=versions[id(store['source'])] or _source_hash(store,limits.block_bytes)!=graph['storages'][sid]['data_sha256']:
                raise StateError('SOURCE_CHANGED_IN_COMMIT_OBSERVER')
        return {**commit,'files':len(files),'storage_bytes':manifest['total_storage_bytes']}
    except BaseException:
        # Retain the failed output as evidence, but revoke this invocation's
        # completion marker even if the final observer failed after publication.
        # This is synchronous failure recovery, not crash atomicity or exclusion
        # of arbitrary concurrent writers.
        (path/'COMMIT.json').unlink(missing_ok=True)
        try:emit(sink,'STATE_INCOMPLETE',path=str(path))
        except BaseException:pass  # Preserve the original save/observer error.
        (path/'COMMIT.json').unlink(missing_ok=True)
        raise

def _read(path,limit):
    if path.is_symlink():raise StateError('SYMLINK_REJECTED')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise StateError('FILE_SIZE_LIMIT')
    return raw

def _json(raw):
    def unique(pairs):
        out={}
        for key,val in pairs:
            if key in out:raise StateError('DUPLICATE_JSON_KEY')
            out[key]=val
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(StateError('NONFINITE_JSON')))

def verify(path,*,expected_manifest_sha256,limits=Limits(),require_commit=True):
    path=Path(path)
    if path.is_symlink() or not path.is_dir():raise StateError('REAL_DIRECTORY_REQUIRED')
    if type(expected_manifest_sha256) is not str or len(expected_manifest_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_manifest_sha256):
        raise StateError('EXTERNAL_MANIFEST_SHA_REQUIRED')
    raw=_read(path/'MANIFEST.json',limits.max_graph_bytes)
    if digest(raw)!=expected_manifest_sha256:raise StateError('MANIFEST_PIN_MISMATCH')
    manifest=_json(raw)
    if manifest.get('format')!=FORMAT or type(manifest.get('files')) is not dict:raise StateError('FORMAT_INVALID')
    total=manifest.get('total_storage_bytes')
    if type(total) is not int or not 0<=total<=limits.max_storage_bytes:raise StateError('STORAGE_LIMIT')
    expected=set(manifest['files'])|{'MANIFEST.json'}
    if require_commit:expected.add('COMMIT.json')
    observed=set()
    for p in path.rglob('*'):
        if p.is_symlink():raise StateError('SYMLINK_REJECTED')
        if p.is_file():observed.add(p.relative_to(path).as_posix())
    if observed!=expected:raise StateError('CHECKPOINT_FILE_SET_MISMATCH')
    for name,ref in manifest['files'].items():
        if type(name) is not str or name.startswith('/') or ':' in name or '\\' in name or '..' in Path(name).parts:
            raise StateError('UNSAFE_BLOCK_PATH')
        data=_read(path/name,limits.max_graph_bytes if name=='STATE.py' else 65536*6+200)
        if len(data)!=ref['bytes'] or digest(data)!=ref['sha256']:raise StateError('CHECKPOINT_FILE_HASH:'+name)
    graphraw=_read(path/'STATE.py',limits.max_graph_bytes)
    if digest(graphraw)!=manifest['graph_sha256']:raise StateError('GRAPH_HASH_MISMATCH')
    graph=_parse(graphraw,'STATE',limits.max_graph_bytes)
    if graph.get('format')!=FORMAT or type(graph.get('nodes')) is not list or len(graph['nodes'])>limits.max_nodes:
        raise StateError('GRAPH_SCHEMA')
    if type(graph.get('storages')) is not list:raise StateError('STORAGE_SCHEMA')
    seen_paths={'STATE.py'};actual_total=0
    for sid,storage in enumerate(graph['storages']):
        if type(storage.get('nbytes')) is not int or storage['nbytes']<0:raise StateError('STORAGE_SCHEMA')
        actual_total+=storage['nbytes'];cursor=0;h=hashlib.sha256()
        if actual_total>limits.max_storage_bytes:raise StateError('STORAGE_LIMIT')
        for chunk in storage['chunks']:
            name=chunk['path']
            if name not in manifest['files'] or name in seen_paths:raise StateError('CHUNK_REFERENCE')
            seen_paths.add(name)
            data=_parse(_read(path/name,65536*6+200),'BLOCK',65536*6+200)
            if type(data.get('data')) is not tuple or any(type(b) is not int or not 0<=b<=255 for b in data['data']):raise StateError('BLOCK_BYTE_VALUE')
            rawbytes=bytes(data['data'])
            if data.get('storage')!=sid or data.get('offset')!=cursor or chunk.get('offset')!=cursor or len(rawbytes)!=chunk['bytes'] or not 0<len(rawbytes)<=65536:
                raise StateError('BLOCK_SEQUENCE')
            if digest(rawbytes)!=chunk['data_sha256']:raise StateError('BLOCK_DATA_HASH')
            h.update(rawbytes);cursor+=len(rawbytes)
        if cursor!=storage['nbytes'] or h.hexdigest()!=storage['data_sha256']:raise StateError('STORAGE_HASH_OR_SIZE')
    if actual_total!=total or set(manifest['files'])!=seen_paths:raise StateError('STORAGE_MANIFEST_CLOSURE')
    if require_commit:
        commit=_json(_read(path/'COMMIT.json',1024))
        if commit!={'format':FORMAT,'status':'COMPLETE','manifest_sha256':expected_manifest_sha256}:raise StateError('COMMIT_MISMATCH')
    return {'status':'LITERAL_STORAGE_BYTES_VERIFIED','graph':graph,'manifest_sha256':expected_manifest_sha256,'storage_bytes':total}

def _decode_graph(graph,storages,*,limits,sink=None):
    """Bind verified plain-data graph to provided stores; native metadata only.

    Shared storage is restored. Live autograd and native version-counter topology
    are not reconstructed. Caller is responsible for full payload custody.
    """
    memo={};active=set()
    def decode(ref,depth=0):
        if depth>limits.max_depth:raise StateError('GRAPH_DEPTH_LIMIT')
        if type(ref) is not dict:raise StateError('REFERENCE_SCHEMA')
        if 'ref' not in ref:
            kind=ref.get('kind')
            if kind=='none' and set(ref)=={'kind'}:return None
            if kind=='bool' and type(ref.get('value')) is bool:return ref['value']
            if kind=='int' and type(ref.get('value')) is str:return int(ref['value'])
            if kind=='str' and type(ref.get('value')) is str:return ref['value']
            if kind=='float64':
                raw=bytes.fromhex(ref['bits_le'])
                if len(raw)==8:return struct.unpack('<d',raw)[0]
            raise StateError('SCALAR_SCHEMA')
        ident=ref['ref']
        if type(ident) is not int or not 0<=ident<len(graph['nodes']):raise StateError('REFERENCE_RANGE')
        if ident in memo:return memo[ident]
        if ident in active:raise StateError('IMMUTABLE_CYCLE_UNSUPPORTED')
        active.add(ident);node=graph['nodes'][ident];kind=node.get('kind');dec=lambda v:decode(v,depth+1)
        if kind=='bytes':result=bytes.fromhex(node['hex'])
        elif kind=='list':
            result=[];memo[ident]=result;result.extend(dec(x) for x in node['items'])
        elif kind=='tuple':result=tuple(dec(x) for x in node['items'])
        elif kind in ('dict','ordered'):
            result=OrderedDict() if kind=='ordered' else {};memo[ident]=result
            for a,b in node['items']:
                key=dec(a);_key_ok(key)
                if key in result:raise StateError('DUPLICATE_DECODED_KEY')
                result[key]=dec(b)
            if kind=='ordered':
                attributes=dec(node['attributes'])
                if type(attributes) is not dict:raise StateError('ATTRIBUTES_SCHEMA')
                result.__dict__.update(attributes)
        elif kind in ('tensor','parameter'):
            dtype=DTYPES.get(node.get('dtype'))
            if dtype is None:raise StateError('TENSOR_DTYPE')
            sid=node['storage'];shape=node['shape'];stride=node['stride'];off=node['offset']
            if type(sid) is not int or not 0<=sid<len(storages):raise StateError('STORAGE_REFERENCE')
            if type(shape) is not list or type(stride) is not list or len(shape)!=len(stride) or any(type(n) is not int or n<0 for n in shape+stride+[off]):raise StateError('TENSOR_GEOMETRY')
            width=torch.empty(0,dtype=dtype).element_size()
            if math.prod(shape) and (off+sum((n-1)*s for n,s in zip(shape,stride))+1)*width>storages[sid].nbytes():raise StateError('TENSOR_STORAGE_BOUNDS')
            gradflag=node['requires_grad']
            if type(gradflag) is not bool:raise StateError('REQUIRES_GRAD_SCHEMA')
            with torch._C._DisableTorchDispatch():
                result=torch.empty(0,dtype=dtype).set_(storages[sid],off,tuple(shape),tuple(stride))
                if kind=='parameter':result=torch.nn.Parameter(result,requires_grad=gradflag)
                elif gradflag:result.requires_grad_(True)
            memo[ident]=result
            attributes=dec(node['attributes'])
            if type(attributes) is not dict:raise StateError('ATTRIBUTES_SCHEMA')
            result.__dict__.update(attributes)
            gradient=dec(node['gradient'])
            if gradient is not None and (type(gradient) is not torch.Tensor or gradient.shape!=result.shape or gradient.dtype!=result.dtype):raise StateError('GRADIENT_SCHEMA')
            result.grad=gradient
            emit(sink,'STATE_TENSOR_BOUND',node=ident,storage=sid,shape=shape,stride=stride,dtype=str(dtype),offset=off,native_binding=True)
        else:raise StateError('NODE_KIND')
        memo[ident]=result;active.remove(ident);return result
    return decode(graph['root'])

def load(path,*,expected_manifest_sha256,limits=Limits(),sink=None):
    receipt=verify(path,expected_manifest_sha256=expected_manifest_sha256,limits=limits)
    graph=receipt['graph'];path=Path(path);storages=[];owners=[]
    for sid,storage in enumerate(graph['storages']):
        raw=bytearray(storage['nbytes'])
        for chunk in storage['chunks']:
            data=_parse(_read(path/chunk['path'],65536*6+200),'BLOCK',65536*6+200)
            payload=bytes(data['data'])
            # Recheck on the actual consumed bytes, not just the previous pass.
            if digest(payload)!=chunk['data_sha256'] or data['offset']!=chunk['offset']:raise StateError('BLOCK_CHANGED_DURING_LOAD')
            raw[chunk['offset']:chunk['offset']+len(payload)]=payload
            emit(sink,'STATE_BLOCK_READ',storage=sid,offset=chunk['offset'],bytes=len(payload),sha256=digest(payload))
        if digest(raw)!=storage['data_sha256']:raise StateError('STORAGE_CHANGED_DURING_LOAD')
        with torch._C._DisableTorchDispatch():
            owner=torch.frombuffer(raw,dtype=torch.uint8) if raw else torch.empty(0,dtype=torch.uint8)
        owners.append(owner);storages.append(owner.untyped_storage())
    result=_decode_graph(graph,storages,limits=limits,sink=sink)
    # Files must still match the independently pinned complete archive.
    verify(path,expected_manifest_sha256=expected_manifest_sha256,limits=limits)
    return result
