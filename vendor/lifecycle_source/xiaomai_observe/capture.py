"""Recursive, explicit tensor capture. Unsupported objects are gaps, not passes."""
from __future__ import annotations
from xiaomai_observe.passive import type_is_one_of
from xiaomai_observe.code_identity import code_fingerprint
from collections.abc import Mapping
import contextlib
import contextvars
import dataclasses
import hashlib
import inspect
import math
import marshal
from pathlib import Path
import threading
import weakref
import torch
from .journal import Journal, digest
from .passive import actual_isinstance, qualified_type, class_attr, static_instance_attr
import collections
import types
import struct
from .readable import TORCH_NAMES,tensor_bytes

BUSY=contextvars.ContextVar('xmo_instrumentation_busy',default=False)

def source_identity(fn):
    f=getattr(fn,'__func__',fn)
    code=getattr(f,'__code__',None)
    result={'module':getattr(f,'__module__',None),'qualname':getattr(f,'__qualname__',None)}
    if code is None:return {**result,'status':'NO_PYTHON_CODE_OBJECT'}
    result.update(file=code.co_filename,first_line=code.co_firstlineno,
                  bytecode_sha256=digest(code.co_code),code_object_sha256=code_fingerprint(code),code_identity_schema='CPYTHON_CODE_STRUCTURAL_VALUES_V1',status='PYTHON_CODE_OBJECT')
    try:
        p=Path(code.co_filename);b=p.read_bytes()
        result['source_file_sha256']=digest(b)
        result['definition_sha256']=digest(inspect.getsource(f).encode('utf-8'))
    except (OSError,TypeError,IOError):
        result['source_available']=False
    return result

@contextlib.contextmanager
def measurement():
    token=BUSY.set(True)
    try:yield
    finally:BUSY.reset(token)


def _numeric_summary(part):
    """Full scan of one chunk; no tensors are retained in the summary."""
    row={'checked_elements':part.numel(),'nan':0,'posinf':0,'neginf':0,'min':None,'max':None}
    if part.is_floating_point():
        row['nan']=int(torch.isnan(part).sum().item())
        row['posinf']=int(torch.isposinf(part).sum().item())
        row['neginf']=int(torch.isneginf(part).sum().item())
        finite=part[torch.isfinite(part)]
    else:finite=part
    if finite.numel():
        row['min']=finite.min().item();row['max']=finite.max().item()
    return row


def _raw_numeric_summary(raw,dtype):
    # Derive the summary from exactly the recorded immutable bytes, never from
    # a mutable source tensor that could change between serialization and scan.
    # One bounded private chunk backs this CPU view; no NumPy/.data alias can
    # mutate it. This preserves BF16/NaN payloads in the actual evidence blob.
    return _numeric_summary(torch.frombuffer(bytearray(raw),dtype=dtype))

class Capture:
    def __init__(self,journal:Journal,*,values='full',numerics=True,chunk_elements=65536):
        if values not in ('full','metadata'):raise ValueError('values must be full or metadata')
        if chunk_elements<=0:raise ValueError('invalid chunk size')
        self.journal=journal;self.values=values;self.numerics=numerics
        self.chunk_elements=chunk_elements
        self.gaps=[];self.tensor_refs={};self.next_id=0;self.lock=threading.RLock()

    def gap(self,reason,**fields):
        row={'reason':reason,**fields};self.gaps.append(row)
        self.journal.emit('OBSERVATION_GAP',**row)
        return {'capture_status':'UNSUPPORTED','reason':reason,**fields}

    def tensor_id(self,t):
        with self.lock:
            key=id(t);old=self.tensor_refs.get(key)
            if old is not None and old[0]() is t:return old[1]
            name=f'tensor:{self.next_id}';self.next_id+=1
            self.tensor_refs[key]=(weakref.ref(t),name)
            return name

    def tensor(self,t,path):
        if not type_is_one_of(type(t),(torch.Tensor,torch.nn.Parameter)):
            return self.gap('TENSOR_SUBCLASS_NOT_PASSIVELY_OBSERVED',path=path,type=qualified_type(type(t)))
        with measurement(),torch.no_grad():
            info={'kind':'tensor','tensor_id':self.tensor_id(t), 'path':path,
                  'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),
                  'requires_grad':t.requires_grad,'layout':str(t.layout),
                  'is_parameter':isinstance(t,torch.nn.Parameter),'total_elements':t.numel()}
            if t.layout!=torch.strided or t.device.type=='meta' or t.is_quantized:
                info.update(self.gap('UNSUPPORTED_TENSOR_LAYOUT_OR_STORAGE',path=path,
                                    dtype=str(t.dtype),layout=str(t.layout)))
                return info
            info.update(stride=list(t.stride()),storage_offset=t.storage_offset(),
                        storage_address=t.untyped_storage().data_ptr(),
                        storage_bytes=t.untyped_storage().nbytes())
            try:info['version_counter']=t._version
            except RuntimeError:info['version_counter']='INFERENCE_TENSOR_NO_VERSION_COUNTER'
            if t.dtype not in TORCH_NAMES:
                info.update(self.gap('UNSUPPORTED_TENSOR_DTYPE',path=path,dtype=str(t.dtype)))
                return info
            if t.numel()==0:
                info['numerics']=({'status':'EMPTY_TENSOR','checked_elements':0,'inspected_elements':0,
                                  'total_elements':0,'sampling_method':'FULL_SCAN','nan':0,'posinf':0,
                                  'neginf':0,'min':None,'max':None} if self.numerics else {'status':'NOT_REQUESTED'})
                info['chunks']=[];info['capture_status']='FULL' if self.values=='full' else 'METADATA_ONLY'
                return info
            # reshape may copy a non-contiguous output. This allocation is explicit
            # in the receipt. It does not keep a graph reference or edit the input.
            view=t.detach()
            info['contiguous_copy_for_capture']=not view.is_contiguous()
            # Flattening a non-contiguous BOOL can copy and canonicalize bytes.
            # Preserve the bytes first, then expose BOOL truth values for scans.
            flat=(view.view(torch.uint8).reshape(-1).view(torch.bool)
                  if view.dtype is torch.bool else view.reshape(-1))
            stats={'checked_elements':0,'nan':0,'posinf':0,'neginf':0,'min':None,'max':None}
            chunks=[];whole=hashlib.sha256();summary_hits=0;summary_scans=0
            for start in range(0,flat.numel(),self.chunk_elements):
                part=flat[start:start+self.chunk_elements]
                if self.values=='full':
                    # Always read every current byte, including when the tensor
                    # version is unchanged (.data/NumPy writes can bypass it).
                    b=tensor_bytes(part);whole.update(b)
                    reference=self.journal.blob(b)
                    chunks.append({'element_start':start,'element_count':part.numel(),**reference})
                if self.numerics:
                    if self.values=='full':
                        row,reused=self.journal.cached_blob_summary(reference,
                            'TENSOR_NUMERICS_CPU_V1:'+TORCH_NAMES[t.dtype],
                            lambda:_raw_numeric_summary(b,t.dtype))
                        summary_hits+=int(reused);summary_scans+=int(not reused)
                    else:
                        row=_numeric_summary(part)
                    for key in ('checked_elements','nan','posinf','neginf'):
                        stats[key]+=row[key]
                    if row['min'] is not None:
                        stats['min']=row['min'] if stats['min'] is None else min(row['min'],stats['min'])
                        stats['max']=row['max'] if stats['max'] is None else max(row['max'],stats['max'])
            if self.numerics:
                stats.update(status='SCANNED',inspected_elements=stats['checked_elements'],
                             total_elements=t.numel(),sampling_method='FULL_SCAN')
                info['numerics']=stats
            else:info['numerics']={'status':'NOT_REQUESTED'}
            info.update(capture_status='FULL' if self.values=='full' else 'METADATA_ONLY')
            if self.values=='full':
                info.update(chunks=chunks,raw_sha256=whole.hexdigest(),raw_dtype=TORCH_NAMES[t.dtype],
                    content_verification={'policy':'FULL_CURRENT_BYTES_WITH_CONTENT_ADDRESSED_SUMMARIES',
                        'bytes_read_and_hashed':t.numel()*t.element_size(),
                        'numeric_summary_chunks_reused':summary_hits,
                        'numeric_summary_chunks_scanned':summary_scans,
                        'version_counter_trusted_for_reuse':False})
            return info

    def tree(self,value,path='value',_ancestors=None):
        with measurement():
            ancestors=set() if _ancestors is None else _ancestors
            ty=type(value)
            if type_is_one_of(ty,(torch.Tensor,torch.nn.Parameter)):return self.tensor(value,path)
            if value is None or type_is_one_of(ty,(bool,int,str)):return value
            if ty is float:
                if math.isnan(value):return {'float':'nan','bits_le':struct.pack('<d',value).hex()}
                if math.isinf(value):return {'float':'+inf' if value>0 else '-inf','bits_le':struct.pack('<d',value).hex()}
                return value
            if ty is bytes:return {'kind':'bytes',**self.journal.blob(value)}
            if id(value) in ancestors:return {'kind':'cycle','object_id':id(value)}
            next_anc=ancestors|{id(value)}
            # Builtin descriptors bypass overridden items/iter/len on subclasses.
            # Arbitrary Mapping implementations are not executed by the observer.
            if actual_isinstance(value,dict):
                pairs=collections.OrderedDict.items(value) if actual_isinstance(value,collections.OrderedDict) else dict.items(value)
                return {'kind':'mapping','type':qualified_type(ty),'items':[
                    [self.tree(k,path+f'.key[{i}]',next_anc),self.tree(v,path+f'.value[{i}]',next_anc)]
                    for i,(k,v) in enumerate(pairs)]}
            if actual_isinstance(value,(list,tuple)):
                seq=list.__iter__(value) if actual_isinstance(value,list) else tuple.__iter__(value)
                return {'kind':'list' if actual_isinstance(value,list) else 'tuple','items':[
                    self.tree(v,path+f'[{i}]',next_anc) for i,v in enumerate(seq)]}
            dc=class_attr(ty,'__dataclass_fields__')
            if type(dc) is dict and not actual_isinstance(value,type):
                fields={}
                for field in dict.values(dc):
                    if type(field) is not dataclasses.Field:
                        return self.gap('NONSTANDARD_DATACLASS_FIELD',path=path,type=qualified_type(ty))
                    if field._field_type is not dataclasses._FIELD:continue
                    if type(field.name) is not str:
                        return self.gap('NONLITERAL_DATACLASS_FIELD_NAME',path=path,type=qualified_type(ty))
                    item=static_instance_attr(value,field.name)
                    if class_attr(type(item),'__get__') is not None:
                        fields[field.name]=self.gap('DATACLASS_DESCRIPTOR_NOT_EXECUTED',path=path+'.'+field.name,type=qualified_type(type(item)))
                    else:fields[field.name]=self.tree(item,path+'.'+field.name,next_anc)
                return {'kind':'dataclass','type':qualified_type(ty),'fields':fields}
            if actual_isinstance(value,torch.nn.Module):
                return {'kind':'module_reference','object_id':id(value),'type':qualified_type(ty)}
            if type_is_one_of(ty,(torch.dtype,torch.device)):
                return {'kind':ty.__name__,'value':str(value)}
            return self.gap('OPAQUE_VALUE_NOT_SERIALIZED',path=path,type=qualified_type(ty))


def model_inventory(model):
    parameters=list(model.named_parameters(remove_duplicate=False))
    buffers=list(model.named_buffers(remove_duplicate=False))
    aliases={}
    for name,p in parameters+buffers:aliases.setdefault(id(p),[]).append(name)
    modules=[]
    for name,m in model.named_modules(remove_duplicate=False):
        modules.append({'name':name,'object_id':id(m),'class':type(m).__module__+'.'+type(m).__qualname__,
                        'forward':source_identity(m.forward)})
    return {'modules':modules,'tensors':[{'name':name,'kind':kind,'object_id':id(p),
        'aliases':aliases[id(p)],'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),
        'requires_grad':p.requires_grad,'elements':p.numel()}
        for kind,rows in [('parameter',parameters),('buffer',buffers)] for name,p in rows],
        'scope':'REGISTERED_MODULES_PARAMETERS_AND_BUFFERS_NOT_ARBITRARY_EXTERNAL_STATE'}
