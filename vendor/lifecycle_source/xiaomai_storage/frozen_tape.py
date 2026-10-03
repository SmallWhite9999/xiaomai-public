"""Explicit immutable CPU parameter references for the bounded Python tape.

Storage is held by the original Tensor; hashing is streamed and never builds
element/offset tuples. Native storage/NumPy views and optional Numba/LLVM are
declared boundaries. Concurrent-writer exclusion is not provided: byte digests
are checked around each consumed operation and at forward/backward boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch
from . import kernels as k, numeric_words as words


class FrozenError(RuntimeError): pass


def metadata(t):
    k.admit(t)
    return (int(t.untyped_storage()._cdata), int(t.untyped_storage().data_ptr()),
            t.untyped_storage().nbytes(), t.dtype, tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), bool(t.requires_grad), t._version)


def storage_hash(t, block_bytes):
    raw=k.raw(t); digest=hashlib.sha256()
    for at in range(0,len(raw),block_bytes):digest.update(raw[at:at+block_bytes])
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenRef:
    owner: object
    tensor: object
    signature: tuple
    shape: tuple
    dtype: object

    def verify(self):
        if metadata(self.tensor)!=self.signature:raise FrozenError('FROZEN_VIEW_METADATA_CHANGED')
        self.owner.verify_metadata()

    def materialize(self, core, limit):
        self.verify()
        if math.prod(self.shape)>limit:raise FrozenError('FROZEN_NON_MATMUL_MATERIALIZATION_LIMIT')
        t=self.tensor;raw=k.raw(t);width=t.element_size()
        values=tuple(words.decode(bytes(raw[(p:=k.offset(t,ix)*width):p+width]),t.dtype)
                     for ix in k.coords(t.shape))
        return core.Array(self.shape,values)


class FrozenStore:
    def __init__(self,t,*,name,block_bytes):
        self.tensor=t;self.names=[name];self.signature=metadata(t)
        self.block_bytes=block_bytes;self.sha256=storage_hash(t,block_bytes)
        self.lazy=False
    def verify_metadata(self):
        if metadata(self.tensor)!=self.signature:raise FrozenError('FROZEN_PARAMETER_METADATA_CHANGED:'+self.names[0])
    def verify(self):
        self.verify_metadata()
        if storage_hash(self.tensor,self.block_bytes)!=self.sha256:
            raise FrozenError('FROZEN_PARAMETER_BYTES_CHANGED:'+self.names[0])


class FrozenParameters:
    def __init__(self,parameters,*,min_elements,block_bytes,emit,reject_hooks):
        if type(parameters) is not dict or any(type(n) is not str or not n for n in parameters):
            raise FrozenError('NAMED_FROZEN_PARAMETER_DICT_REQUIRED')
        self.stores={};self.parameters=[];self.retained=[];self.emit=emit
        self.block_bytes=block_bytes
        for name,t in parameters.items():
            k.admit(t);reject_hooks(t)
            if t.requires_grad or t.dtype not in (torch.float32,torch.float64) or t.grad_fn is not None:
                raise FrozenError('FROZEN_CPU_F32_F64_NONTRAINABLE_REQUIRED:'+name)
            signature=metadata(t);key=signature[0]
            if key not in self.stores:
                self.stores[key]=FrozenStore(t,name=name,block_bytes=block_bytes)
            else:
                store=self.stores[key]
                if store.signature[3] is not t.dtype:raise FrozenError('FROZEN_SHARED_STORAGE_DTYPE_MISMATCH')
                store.names.append(name)
            store=self.stores[key]
            store.lazy=store.signature[2]//t.element_size()>=min_elements
            self.parameters.append((name,t,signature))
        self.verify_all()
        self.emit('PY_FROZEN_PARAMETER_POLICY',parameters=len(self.parameters),storages=len(self.stores),
                  lazy_storages=sum(s.lazy for s in self.stores.values()),
                  storage_bytes=sum(s.signature[2] for s in self.stores.values()),
                  hash_block_bytes=block_bytes,concurrent_writer_exclusion=False,
                  immutable_storage_no_element_tuple=True)
    def store(self,t):
        return self.stores.get(int(t.untyped_storage()._cdata))
    def ref(self,t):
        store=self.store(t)
        if store is None or not store.lazy:return None
        store.verify_metadata();signature=metadata(t)
        if signature[3] is not store.signature[3] or t.requires_grad:
            raise FrozenError('FROZEN_ALIAS_DTYPE_OR_TRAINABILITY_CHANGED')
        self.retained.append(t)
        return FrozenRef(store,t,signature,tuple(t.shape),t.dtype)
    def verify_all(self):
        for name,t,signature in self.parameters:
            if metadata(t)!=signature:raise FrozenError('FROZEN_PARAMETER_IDENTITY_CHANGED:'+name)
        for store in self.stores.values():store.verify()
    def touched(self,values):
        result={}
        for t in values:
            store=self.store(t)
            if store is not None:
                if t.dtype is not store.signature[3]:raise FrozenError('FROZEN_ALIAS_DTYPE_CHANGED')
                result[store.signature[0]]=store
        return tuple(result.values())


def multiply(core,left,right,*,transpose_left=False,transpose_right=False,
             batch=False,mode='python',emit=None,record_values=False):
    """VJP product only: no native matmul/backward and no frozen-weight copy."""
    from .cpu_compiled import array,compiled
    from xiaomai_python import streaming_mm
    import numpy as np
    def tensor(value,transpose):
        if type(value) is FrozenRef:
            value.verify();t=value.tensor
        else:
            # This is already budgeted tape/adjoint data, never the large
            # frozen parameter. NumPy and Torch provide storage binding only.
            t=torch.from_numpy(np.asarray(value.data,dtype=np.float64).reshape(value.shape))
        return t.transpose(-1,-2) if transpose else t
    with torch._C._DisableTorchDispatch(),torch.no_grad():
        a=tensor(left,transpose_left);b=tensor(right,transpose_right)
        if emit:emit('PY_FROZEN_VJP_PRODUCT_BEGIN',shape_a=list(a.shape),shape_b=list(b.shape),
                     implementation='COMPILED_PYTHON_BINARY64' if mode=='compiled' else 'INTERPRETED_PYTHON_BINARY64',
                     native_blas=False,native_backward=False,scalar_instruction_trace=mode=='python' and record_values,
                     fastmath=False)
        if mode=='compiled':
            product=compiled('bmm' if batch else 'mm')(array(a),array(b))
            result=core.Array(tuple(product.shape),tuple(float(x) for x in product.flat))
        else:
            fn=streaming_mm.bmm if batch else streaming_mm.mm
            result=fn(a,b,record=emit if record_values else None)
        if emit:emit('PY_FROZEN_VJP_PRODUCT_END',shape=list(result.shape),elements=len(result.data))
        return result
