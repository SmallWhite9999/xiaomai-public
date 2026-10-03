"""Python allocation geometry, admission and byte-buffer ownership.

The default 'native-compatible' strategy preserves PyTorch resizable storage.
The explicitly selected 'python-buffer' strategy owns zero-initialized bytearray
storage and binds it with torch.frombuffer/as_strided. That storage is NOT
resizable and is not advertised as fully equivalent to the native allocator.
No pinning or GPU capability is fabricated. Allocation callbacks are scoped with
ContextVar; original framework allocator and private data are not replaced.
"""
from __future__ import annotations
import contextvars,math,sys,threading
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from . import kernels as k

OPS=frozenset(('aten.empty.memory_format','aten.empty_strided.default','aten.empty_like.default',
               'aten.new_empty.default','aten.new_empty_strided.default'))
WIDTHS={torch.bool:1,torch.uint8:1,torch.int8:1,torch.int16:2,torch.int32:4,torch.int64:8,
        torch.float16:2,torch.bfloat16:2,torch.float32:4,torch.float64:8}
_CURRENT=contextvars.ContextVar('xiaomai_python_allocator',default=None)

@dataclass(frozen=True)
class Plan:
    shape:tuple
    stride:tuple
    dtype:object
    nbytes:int
    logical_elements:int

def plan(shape,stride,*,dtype):
    shape=k.shape_tuple(shape);stride=k.shape_tuple(stride)
    if len(shape)!=len(stride):raise k.StorageError('ALLOCATION_RANK_MISMATCH')
    if not any(dtype is d for d in k.DTYPES):raise k.StorageError('ALLOCATION_DTYPE')
    if any(n>sys.maxsize for n in shape+stride):raise k.StorageError('ALLOCATION_METADATA_OVERFLOW')
    numel=math.prod(shape)
    if numel>sys.maxsize:raise k.StorageError('ALLOCATION_ELEMENT_OVERFLOW')
    count=0 if numel==0 else 1+sum((n-1)*s for n,s in zip(shape,stride))
    size=count*WIDTHS[dtype]
    if size>sys.maxsize:raise k.StorageError('ALLOCATION_STORAGE_OVERFLOW')
    return Plan(shape,stride,dtype,size,numel)

def current():return _CURRENT.get()

class AllocationScope:
    def __init__(self,*,strategy='native-compatible',sink=None,max_single_bytes=2**30,max_total_bytes=4*2**30):
        if strategy not in ('native-compatible','python-buffer'):raise k.StorageError('ALLOCATION_STRATEGY')
        if any(type(v) is not int or v<=0 for v in (max_single_bytes,max_total_bytes)):
            raise k.StorageError('ALLOCATION_POSITIVE_LIMITS')
        self.strategy=strategy;self.sink=sink;self.max_single=max_single_bytes;self.max_total=max_total_bytes
        self.total=0;self.count=0;self.failed=False;self.token=None
    def emit(self,kind,**fields):
        if self.sink is not None:self.sink(kind,**fields)
    def __enter__(self):
        if current() is not None:raise k.StorageError('NESTED_ALLOCATION_SCOPE_NOT_ADMITTED')
        self.owner=threading.get_ident();self.token=_CURRENT.set(self)
        try:self.emit('ALLOCATION_POLICY',strategy=self.strategy,python_geometry=True,
                  native_tensor_binding=True,storage_resizable=self.strategy=='native-compatible',
                  python_buffer_empty_initialization='ZERO; unspecified native empty values are not compared',
                  limit_policy='cumulative allocated bytes, not current RSS',
                  max_single_bytes=self.max_single,max_total_bytes=self.max_total)
        except BaseException:
            _CURRENT.reset(self.token);self.failed=True;raise
        return self
    def __exit__(self,*args):
        _CURRENT.reset(self.token);self.failed|=args[0] is not None
        self.emit('ALLOCATION_SCOPE_END',calls=self.count,cumulative_bytes=self.total,failed=self.failed)
    def allocate(self,p:Plan):
        if threading.get_ident()!=self.owner:raise k.StorageError('ALLOCATION_THREAD_MISMATCH')
        if p.nbytes>self.max_single or self.total+p.nbytes>self.max_total:
            self.failed=True;self.emit('ALLOCATION_REJECTED',bytes=p.nbytes,reason='LIMIT_BEFORE_ALLOCATION')
            raise k.StorageError('ALLOCATION_RESOURCE_LIMIT')
        ident=self.count;self.count+=1
        self.emit('ALLOCATION_BEGIN',allocation=ident,shape=list(p.shape),stride=list(p.stride),dtype=str(p.dtype),bytes=p.nbytes)
        try:
            with torch._C._DisableTorchDispatch(),torch.no_grad():
                if self.strategy=='native-compatible' or p.nbytes==0:
                    result=torch.empty_strided(p.shape,p.stride,dtype=p.dtype,device='cpu')
                    binding='native_empty_strided' if p.nbytes else 'zero_byte_native_metadata'
                else:
                    owner=bytearray(p.nbytes)
                    tensor=torch.frombuffer(owner,dtype=p.dtype)
                    result=torch.as_strided(tensor,p.shape,p.stride,0)
                    binding='Python_bytearray_to_native_frombuffer_as_strided'
            self.total+=p.nbytes
            self.emit('ALLOCATION_END',allocation=ident,binding=binding,bytes=p.nbytes,
                      strategy=self.strategy,storage_address=result.untyped_storage().data_ptr())
            return result
        except BaseException:
            self.failed=True;self.emit('ALLOCATION_ERROR',allocation=ident);raise

def allocate(shape,stride,*,dtype):
    p=plan(shape,stride,dtype=dtype);scope=current()
    if scope is not None:return scope.allocate(p)
    with torch._C._DisableTorchDispatch(),torch.no_grad():
        return torch.empty_strided(p.shape,p.stride,dtype=p.dtype,device='cpu')

def execute(name,v,*,sink=None,progress=None):
    if name not in OPS:raise k.StorageError('ALLOCATION_OPERATOR_UNKNOWN')
    source=v.get('self')
    if source is not None:k.admit(source)
    dtype=v.get('dtype') or (source.dtype if source is not None else torch.get_default_dtype())
    if v.get('device') is not None and torch.device(v['device']).type!='cpu':raise k.StorageError('CPU_ONLY')
    if v.get('layout') is not None and v['layout'] is not torch.strided:raise k.StorageError('STRIDED_ONLY')
    if v.get('pin_memory') is True:raise k.StorageError('PINNING_REQUIRES_NATIVE_DEVICE_SERVICE')
    shape=tuple(source.shape) if name=='aten.empty_like.default' else k.shape_tuple(v['size'])
    stride=k.shape_tuple(v['stride']) if 'strided' in name else k.output_strides(shape,
        like=source if name=='aten.empty_like.default' else None,memory_format=v.get('memory_format'))
    p=plan(shape,stride,dtype=dtype)
    k.emit(sink,'PYTHON_ALLOCATION_PLAN',operator=name,shape=list(shape),stride=list(stride),
           bytes=p.nbytes,policy='native-compatible' if current() is None else current().strategy,
           native_binding_remains=True)
    return allocate(shape,stride,dtype=dtype)
