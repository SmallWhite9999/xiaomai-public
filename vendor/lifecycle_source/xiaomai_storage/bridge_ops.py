"""CPU geometry/cast bridges on existing Tensor objects.

Python decides size equality, storage byte extents, dtype/copy policy and view
geometry. Alias creation/version/autograd wrapping remains a declared native
primitive. This file never advertises those runtime internals as Python code.
"""
from __future__ import annotations
import math
import torch
from . import kernels as k, numeric_words as words, view_ops

DATA_OPERATORS=frozenset(('aten.contiguous.default','aten.to.dtype','aten.to.device',
    'aten.lift_fresh_copy.default','aten.equal.default','aten.is_same_size.default','aten._has_same_storage_numel.default'))
VIEW_OPERATORS=frozenset(('aten._unsafe_view.default','aten.as_strided.default','aten.detach.default'))
OPERATORS=DATA_OPERATORS|VIEW_OPERATORS

def _same_layout(t,memory_format):
    # Native is_contiguous is metadata-only. Compute the non-singleton stride
    # relations here, rather than delegating the actual contiguous copy kernel.
    if t.numel()==0:return True
    if memory_format is torch.contiguous_format or memory_format is None:
        order=tuple(reversed(range(t.ndim)))
    elif memory_format is torch.channels_last and t.ndim==4:order=(1,3,2,0)
    elif memory_format is torch.channels_last_3d and t.ndim==5:order=(1,4,3,2,0)
    else:raise k.StorageError('BRIDGE_MEMORY_FORMAT_NOT_ADMITTED')
    expected=1
    for axis in order:
        if t.shape[axis]!=1:
            if t.stride()[axis]!=expected:return False
            expected*=t.shape[axis]
    return True

def _validate_geometry(t,size,stride,offset):
    size=view_ops.integer_sequence(size);stride=view_ops.integer_sequence(stride)
    if len(size)!=len(stride) or type(offset) is not int or offset<0:
        raise k.StorageError('AS_STRIDED_METADATA_INVALID')
    if any(n<0 for n in size) or any(n<0 for n in stride):
        raise k.StorageError('AS_STRIDED_NEGATIVE_METADATA')
    if any(n>2**63-1 for n in (*size,*stride,offset)):
        raise k.StorageError('AS_STRIDED_METADATA_OVERFLOW')
    # Empty tensors permit offsets outside storage, consistent with PyTorch.
    if math.prod(size):
        end=(offset+sum((n-1)*s for n,s in zip(size,stride))+1)*t.element_size()
        if end>t.untyped_storage().nbytes():raise k.StorageError('AS_STRIDED_STORAGE_BOUNDS')
    return size,stride,offset

def execute(name,v,*,sink=None,progress=None):
    t=k.admit(v['self'])
    if name=='aten.equal.default':
        other=k.admit(v['other'])
        if tuple(t.shape)!=tuple(other.shape):return False
        if t.dtype is not other.dtype:raise k.StorageError('EQUAL_MIXED_DTYPE_NOT_ADMITTED')
        left,right=k.raw(t),k.raw(other);width=t.element_size()
        for ix in k.coords(t.shape):
            a=k.offset(t,ix)*width;b=k.offset(other,ix)*width
            if words.decode(bytes(left[a:a+width]),t.dtype)!=words.decode(bytes(right[b:b+width]),other.dtype):return False
        return True
    if name=='aten.is_same_size.default':
        other=k.admit(v['other']);return tuple(t.shape)==tuple(other.shape)
    if name=='aten._has_same_storage_numel.default':
        other=k.admit(v['other'])
        # ATen compares storage element capacity for each dtype, not logical
        # shape/numel and not alias identity.
        return t.untyped_storage().nbytes()//t.element_size()==other.untyped_storage().nbytes()//other.element_size()
    if name=='aten.lift_fresh_copy.default':
        return k.clone(t,sink=sink,progress=progress)
    if name=='aten.contiguous.default':
        fmt=v.get('memory_format',torch.contiguous_format)
        if _same_layout(t,fmt):return t
        return k.clone(t,memory_format=fmt,sink=sink,progress=progress)
    if name in ('aten.to.dtype','aten.to.device'):
        if v.get('non_blocking',False):raise k.StorageError('NONBLOCKING_CAST_NOT_ADMITTED')
        if v.get('device') is not None and torch.device(v['device']).type!='cpu':raise k.StorageError('CPU_ONLY')
        dtype=v['dtype'];fmt=v.get('memory_format')
        matches=fmt is None or fmt is torch.preserve_format or _same_layout(t,fmt)
        if dtype is t.dtype and matches and not v.get('copy',False):return t
        return words.execute('aten._to_copy.default',{
            'self':t,'dtype':dtype,'device':'cpu','memory_format':fmt,'non_blocking':False,
            'layout':torch.strided,'pin_memory':False},sink=sink,progress=progress)
    if name=='aten._unsafe_view.default':
        shape=view_ops.inferred_shape(v['size'],t.numel())
        stride=view_ops.view_strides(tuple(t.shape),tuple(t.stride()),shape)
        k.emit(sink,'ALIAS_SEMANTICS',operation=name,
               native_primitive='as_strided constructor + outer UnsafeView autograd wrapper',
               python_geometry=True,native_version_and_autograd=True)
        return view_ops.new_view(t,shape,stride,sink=sink,operation=name)
    if name=='aten.as_strided.default':
        offset=t.storage_offset() if v.get('storage_offset') is None else v['storage_offset']
        shape,stride,offset=_validate_geometry(t,v['size'],v['stride'],offset)
        return view_ops.new_view(t,shape,stride,offset,sink=sink,operation=name)
    if name=='aten.detach.default':
        k.emit(sink,'ALIAS_SEMANTICS',operation=name,python_geometry=True,
               native_primitive='as_strided constructor + outer detach/autograd wrapper',
               native_version_and_autograd=True)
        return view_ops.new_view(t,tuple(t.shape),tuple(t.stride()),sink=sink,operation=name)
    raise k.StorageError('BRIDGE_OPERATION_UNKNOWN')
