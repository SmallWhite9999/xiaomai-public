"""CPU storage kernels with Python-owned offsets and bounded byte movement.

No clone/copy_/zeros/ones/full/fill/eye data kernel is called here.
BOOL copy follows native truth canonicalization (0/1); source bytes are logged.
Native dependencies: CPU allocation, tensor metadata, CPython raw-memory access,
scalar dtype conversion, and the original dispatcher/autograd scheduling.
Use StorageBackend, not these low-level functions, for autograd/version semantics.
"""
from __future__ import annotations
import ctypes
import itertools
import math
from typing import Callable
import torch

BLOCK_BYTES = 65536
DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
          torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64)

class StorageError(RuntimeError):
    pass

def admit(t):
    if type(t) is not torch.Tensor and type(t) is not torch.nn.Parameter:
        raise StorageError("EXACT_TENSOR_OR_PARAMETER_REQUIRED")
    if (t.device.type != "cpu" or t.layout is not torch.strided or t.is_quantized
            or t.is_complex() or t.is_conj() or t.is_neg()):
        raise StorageError("CPU_REAL_PLAIN_STRIDED_REQUIRED")
    if not any(t.dtype is d for d in DTYPES):
        raise StorageError("DTYPE_NOT_ADMITTED")
    if any(s < 0 for s in t.stride()):
        raise StorageError("NEGATIVE_STRIDE_NOT_ADMITTED")
    nbytes = t.untyped_storage().nbytes()
    if t.numel():
        highest = t.storage_offset() + sum((n-1)*s for n,s in zip(t.shape,t.stride()))
        if t.storage_offset() < 0 or (highest+1)*t.element_size() > nbytes:
            raise StorageError("STORAGE_BOUNDS")
    return t

def shape_tuple(shape):
    if type(shape) is not tuple and type(shape) is not list and type(shape) is not torch.Size:
        raise StorageError("CONCRETE_SHAPE_REQUIRED")
    if any(type(n) is not int or n < 0 for n in shape):
        raise StorageError("NONNEGATIVE_INTEGER_SHAPE_REQUIRED")
    return tuple(shape)

def coords(shape):
    return itertools.product(*(range(n) for n in shape))

def offset(t, ix):
    return t.storage_offset() + sum(i*s for i,s in zip(ix,t.stride()))

def raw(t):
    admit(t)
    storage=t.untyped_storage()
    if storage.nbytes()==0:
        return memoryview(bytearray())
    return memoryview((ctypes.c_ubyte*storage.nbytes()).from_address(storage.data_ptr())).cast("B")

def logical_bytes(t):
    """Diagnostic helper: complete logical byte order, not used for bulk kernels."""
    admit(t); v=raw(t); width=t.element_size()
    return b"".join(bytes(v[offset(t,ix)*width:(offset(t,ix)+1)*width]) for ix in coords(t.shape))

def nonoverlap(t):
    if t.numel()==0: return True
    required=1
    for stride, size in sorted((s,n) for s,n in zip(t.stride(),t.shape) if n>1):
        if stride < required: return False
        required += (size-1)*stride
    return True

def dense(t):
    if t.numel()==0: return True
    expected=1
    for stride,size in sorted((s,n) for s,n in zip(t.stride(),t.shape) if n>1):
        if stride != expected:return False
        expected *= size
    return True

def contiguous_strides(shape, order=None):
    order=tuple(reversed(range(len(shape)))) if order is None else tuple(order)
    strides=[0]*len(shape); step=1
    for axis in order:
        strides[axis]=step
        step*=max(1,shape[axis])
    return tuple(strides)

def output_strides(shape, *, like=None, memory_format=None):
    if memory_format is torch.preserve_format or memory_format is None:
        if like is None:return contiguous_strides(shape)
        if dense(like): return tuple(like.stride())
        # At this boundary only monotone strided or overlapping contiguous-order
        # inputs are admitted. Unusual non-dense permutations are rejected.
        active=[(s,n) for s,n in zip(like.stride(),shape) if n>1 and s>0]
        if any(a[0]<b[0] for a,b in zip(active,active[1:])):
            raise StorageError("NON_DENSE_PERMUTED_LAYOUT_NOT_YET_ADMITTED")
        # Preserve the native dense output's physical dimension ordering,
        # including size-one axes. A plain contiguous fallback changes stride
        # metadata when equal source strides order singleton axes first.
        order=list(reversed(range(len(shape))))
        source_strides=tuple(like.stride())
        for i in range(1,len(order)):
            at=i
            for previous in reversed(range(i)):
                a,b=order[previous],order[at]
                sa,sb=source_strides[a],source_strides[b]
                if sa==0 or sb==0:continue
                if sa>sb or sa==sb and shape[a]>shape[b]:
                    order[previous],order[at]=order[at],order[previous]
                    at=previous
                elif sa<sb:break
        return contiguous_strides(shape,order)
    if memory_format is torch.contiguous_format:return contiguous_strides(shape)
    if memory_format is torch.channels_last and len(shape)==4:
        return contiguous_strides(shape,(1,3,2,0))
    if memory_format is torch.channels_last_3d and len(shape)==5:
        return contiguous_strides(shape,(1,4,3,2,0))
    raise StorageError("MEMORY_FORMAT_NOT_ADMITTED")

def allocate(shape, *, dtype, like=None, memory_format=None, device=None,
             layout=None, pin_memory=None):
    shape=shape_tuple(shape)
    if not any(dtype is d for d in DTYPES):raise StorageError("DTYPE_NOT_ADMITTED")
    if device is not None and torch.device(device).type!="cpu":raise StorageError("CPU_ONLY")
    if layout is not None and layout is not torch.strided:raise StorageError("STRIDED_ONLY")
    if pin_memory not in (None,False):raise StorageError("PIN_MEMORY_NOT_ADMITTED")
    strides=output_strides(shape,like=like,memory_format=memory_format)
    # Strategy is explicit and scoped. Default preserves native resizable storage.
    from .allocation import allocate as allocate_storage
    return allocate_storage(shape,strides,dtype=dtype)

def scalar_bits(value,dtype):
    if not any(type(value) is t for t in (int,float,bool)):raise StorageError("BUILTIN_SCALAR_REQUIRED")
    from .numeric_words import encode_scalar
    return encode_scalar(value,dtype,strict_integer=True)

def emit(sink, kind, **fields):
    if sink is not None:sink(kind,**fields)

def tick(progress, **fields):
    if progress is not None:progress(**fields)

def fill(target, pattern, *, sink=None, progress=None, operation="fill", fresh=False):
    admit(target)
    if len(pattern)!=target.element_size():raise StorageError("PATTERN_WIDTH")
    dst=raw(target); width=target.element_size()
    # Repeated writes to expanded storage are safe for a constant fill.
    if dense(target):
        total=target.numel()*width; first=target.storage_offset()*width
        for relative in range(0,total,BLOCK_BYTES):
            size=min(BLOCK_BYTES,total-relative); at=first+relative
            tick(progress,operation=operation,byte_offset=relative,byte_count=size)
            old=None if fresh else bytes(dst[at:at+size]).hex()
            block=pattern*(size//width)
            emit(sink,"STORAGE_WRITE_INTENT",operation=operation,dst_byte=at,
                 before_hex=old,after_hex=block.hex(),fresh_allocation=fresh)
            dst[at:at+size]=block
            emit(sink,"STORAGE_WRITE_DONE",operation=operation,dst_byte=at,byte_count=size)
    else:
        for ordinal,ix in enumerate(coords(target.shape)):
            if ordinal % 4096==0:tick(progress,operation=operation,element=ordinal)
            at=offset(target,ix)*width;old=None if fresh else bytes(dst[at:at+width]).hex()
            emit(sink,"STORAGE_WRITE_INTENT",operation=operation,dst_byte=at,
                 before_hex=old,after_hex=pattern.hex(),fresh_allocation=fresh)
            dst[at:at+width]=pattern
            emit(sink,"STORAGE_WRITE_DONE",operation=operation,dst_byte=at,byte_count=width)
    return target

def extent(t):
    if not t.numel():return (0,0)
    lo=t.untyped_storage().data_ptr()+t.storage_offset()*t.element_size()
    hi=lo+(1+sum((n-1)*s for n,s in zip(t.shape,t.stride())))*t.element_size()
    return lo,hi

def copy(source,target,*,sink=None,progress=None,operation="copy",fresh=False):
    admit(source);admit(target)
    if source.dtype is not target.dtype or tuple(source.shape)!=tuple(target.shape):
        from .numeric_words import copy as copy_converting
        return copy_converting(source,target,sink=sink,progress=progress,operation=operation,fresh=fresh)
    if not nonoverlap(target):raise StorageError("OVERLAPPING_DESTINATION")
    a,b=extent(source);c,d=extent(target)
    same=(a==c and tuple(source.stride())==tuple(target.stride()))
    if a<d and c<b and not same:raise StorageError("OVERLAPPING_SOURCE_DESTINATION")
    src,dst=raw(source),raw(target);width=source.element_size()
    if same:
        emit(sink,"STORAGE_SELF_COPY",operation=operation,elements=source.numel())
        return target
    if not source.numel():return target
    shape=tuple(source.shape);last=shape[-1] if shape else 1
    outer=coords(shape[:-1]) if shape else [()]
    for prefix in outer:
        base=prefix+(0,) if shape else ()
        ss=source.stride()[-1] if shape else 1
        ts=target.stride()[-1] if shape else 1
        first_src=offset(source,base)*width;first_dst=offset(target,base)*width
        if ss==ts==1:
            for first in range(0,last,max(1,BLOCK_BYTES//width)):
                count=min(BLOCK_BYTES//width,last-first);length=count*width
                sa=first_src+first*width;da=first_dst+first*width
                tick(progress,operation=operation,row=list(prefix),element=first,count=count)
                source_block=bytes(src[sa:sa+length])
                block=(bytes(1 if b else 0 for b in source_block) if source.dtype is torch.bool else source_block)
                old=None if fresh else bytes(dst[da:da+length]).hex()
                emit(sink,"STORAGE_WRITE_INTENT",operation=operation,src_byte=sa,
                     dst_byte=da,before_hex=old,source_hex=source_block.hex(),after_hex=block.hex(),fresh_allocation=fresh)
                dst[da:da+length]=block
                emit(sink,"STORAGE_WRITE_DONE",operation=operation,dst_byte=da,byte_count=length)
        else:
            for j in range(last):
                if j%4096==0:tick(progress,operation=operation,row=list(prefix),element=j)
                sa=first_src+j*ss*width;da=first_dst+j*ts*width
                source_block=bytes(src[sa:sa+width])
                block=(bytes(1 if b else 0 for b in source_block) if source.dtype is torch.bool else source_block)
                old=None if fresh else bytes(dst[da:da+width]).hex()
                emit(sink,"STORAGE_WRITE_INTENT",operation=operation,src_byte=sa,
                     dst_byte=da,before_hex=old,source_hex=source_block.hex(),after_hex=block.hex(),fresh_allocation=fresh)
                dst[da:da+width]=block
                emit(sink,"STORAGE_WRITE_DONE",operation=operation,dst_byte=da,byte_count=width)
    return target

def clone(source, memory_format=None, *,sink=None,progress=None):
    admit(source)
    result=allocate(tuple(source.shape),dtype=source.dtype,like=source,memory_format=memory_format)
    return copy(source,result,sink=sink,progress=progress,operation="clone",fresh=True)

def infer_dtype(value):
    if type(value) is bool:return torch.bool
    if type(value) is int:return torch.int64
    if type(value) is float:return torch.get_default_dtype()
    raise StorageError("BUILTIN_SCALAR_REQUIRED")

def factory(shape, value, *, dtype=None, like=None, memory_format=None,
            device=None,layout=None,pin_memory=None,sink=None,progress=None,
            operation="factory"):
    if like is not None:admit(like)
    dtype=dtype if dtype is not None else (like.dtype if like is not None else infer_dtype(value))
    # Validate the scalar before allocating a large result.
    bits=scalar_bits(value,dtype)
    result=allocate(shape,dtype=dtype,like=like,memory_format=memory_format,
                    device=device,layout=layout,pin_memory=pin_memory)
    return fill(result,bits,sink=sink,progress=progress,operation=operation,fresh=True)

def eye(n,m=None,*,dtype=None,device=None,layout=None,pin_memory=None,sink=None,progress=None):
    n,m=shape_tuple((n,n if m is None else m))
    dtype=torch.get_default_dtype() if dtype is None else dtype
    result=factory((n,m),0,dtype=dtype,device=device,layout=layout,pin_memory=pin_memory,
                   sink=sink,progress=progress,operation="eye_zero")
    dst=raw(result);one=scalar_bits(1,dtype);width=result.element_size()
    for j in range(min(n,m)):
        if j%4096==0:tick(progress,operation="eye_diagonal",element=j)
        at=(j*m+j)*width
        emit(sink,"STORAGE_WRITE_INTENT",operation="eye_diagonal",dst_byte=at,
             before_hex=bytes(dst[at:at+width]).hex(),after_hex=one.hex(),fresh_allocation=False)
        dst[at:at+width]=one
        emit(sink,"STORAGE_WRITE_DONE",operation="eye_diagonal",dst_byte=at,byte_count=width)
    return result

OPERATORS=frozenset("aten."+name for name in (
    "clone.default","copy_.default","zeros.default","zeros_like.default",
    "ones.default","ones_like.default","eye.default","eye.m",
    "full.default","full_like.default","new_zeros.default","new_full.default",
    "zero_.default","fill_.Scalar","fill_.Tensor"))

def execute(name,v,*,sink=None,progress=None):
    if name not in OPERATORS:raise StorageError("OPERATOR_NOT_IMPLEMENTED:"+name)
    kw=dict(sink=sink,progress=progress)
    if name=="aten.clone.default":
        return clone(v["self"],v.get("memory_format"),**kw)
    if name=="aten.copy_.default":
        if v.get("non_blocking",False):raise StorageError("NONBLOCKING_COPY_NOT_ADMITTED")
        return copy(v["src"],v["self"],operation=name,**kw)
    if name in ("aten.zero_.default","aten.fill_.Scalar","aten.fill_.Tensor"):
        target=admit(v["self"]);value=0 if name=="aten.zero_.default" else v["value"]
        if name=="aten.fill_.Tensor":
            admit(value)
            if value.ndim!=0:raise StorageError("FILL_ZERO_DIM_SCALAR_REQUIRED")
            if value.dtype is not target.dtype:raise StorageError("FILL_TENSOR_CAST_NOT_ADMITTED")
            source_pattern=logical_bytes(value)
            pattern=bytes(1 if b else 0 for b in source_pattern) if target.dtype is torch.bool else source_pattern
            emit(sink,"SCALAR_FILL_VALUE",operation=name,source_hex=source_pattern.hex(),
                 resulting_pattern_hex=pattern.hex(),dtype=str(target.dtype))
        else:pattern=scalar_bits(value,target.dtype)
        return fill(target,pattern,operation=name,**kw)
    opts={k:v.get(k) for k in ("dtype","device","layout","pin_memory")}
    if name in ("aten.eye.default","aten.eye.m"):
        return eye(v["n"],v.get("m"),**opts,**kw)
    like=v.get("self") if "like" in name or ".new_" in name else None
    shape=tuple(like.shape) if "_like." in name else v["size"]
    value=v.get("fill_value",1 if "ones" in name else 0)
    # zeros/ones default to floating dtype, unlike full(size, int_scalar).
    if opts["dtype"] is None and like is None and "full" not in name:
        opts["dtype"]=torch.get_default_dtype()
    template=like if "_like." in name else None
    if like is not None and opts["dtype"] is None:opts["dtype"]=like.dtype
    return factory(shape,value,like=template,memory_format=v.get("memory_format"),
                   operation=name,**opts,**kw)
