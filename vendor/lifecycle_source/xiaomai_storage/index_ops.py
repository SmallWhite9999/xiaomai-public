"""Python coordinate-based data movement and scalar selection; no native index kernels."""
from __future__ import annotations
import struct
import torch
from . import kernels as k

OPERATORS=frozenset("aten."+n for n in (
    "cat.default","stack.default","flip.default","index_select.default",
    "_local_scalar_dense.default","is_nonzero.default","nonzero.default"))


def require_canonical_bool(t):
    if t.dtype is not torch.bool:return
    view=k.raw(t)
    for ix in k.coords(t.shape):
        if view[k.offset(t,ix)] not in (0,1):
            # Native BOOL raw-byte normalization differs by fast path/layout.
            # Keep original evidence bytes; refuse instead of guessing a kernel.
            raise k.StorageError("NONCANONICAL_BOOL_INDEX_KERNEL_NOT_ADMITTED")

def dimension(dim,rank,insert=False):
    upper=rank+1 if insert else rank
    if type(dim) is not int or not -upper<=dim<upper:
        raise k.StorageError("DIMENSION_OUT_OF_RANGE")
    return dim%upper

def decode_word(bits,dtype):
    if dtype is torch.bool:return bits[0]!=0
    codes={torch.uint8:"B",torch.int8:"b",torch.int16:"h",torch.int32:"i",torch.int64:"q",
           torch.float16:"e",torch.float32:"f",torch.float64:"d"}
    if dtype is torch.bfloat16:return struct.unpack("<f",b"\0\0"+bits)[0]
    code=next((fmt for dt,fmt in codes.items() if dt is dtype),None)
    if code is None:raise k.StorageError("DTYPE_NOT_ADMITTED")
    return struct.unpack("<"+code,bits)[0]

def value_at(t,ix):
    view=k.raw(t);a=k.offset(t,ix)*t.element_size()
    return decode_word(bytes(view[a:a+t.element_size()]),t.dtype)

def mapped_copy(src,dst,map_index,*,sink=None,progress=None,operation):
    source,target=k.raw(src),k.raw(dst);width=src.element_size()
    for ordinal,ix in enumerate(k.coords(src.shape)):
        if ordinal%4096==0:k.tick(progress,operation=operation,element=ordinal)
        si=k.offset(src,ix)*width;di=k.offset(dst,map_index(ix))*width
        bits=bytes(source[si:si+width])
        out=bytes(1 if b else 0 for b in bits) if src.dtype is torch.bool else bits
        k.emit(sink,"STORAGE_WRITE_INTENT",operation=operation,src_byte=si,dst_byte=di,
               source_hex=bits.hex(),before_hex=None,after_hex=out.hex(),fresh_allocation=True)
        target[di:di+width]=out
        k.emit(sink,"STORAGE_WRITE_DONE",operation=operation,dst_byte=di,byte_count=width)

def suggested_format(source):
    """Python stride rule matching the admitted CPU cat layout selection."""
    if source.ndim==4:order=(1,3,2,0);result=torch.channels_last
    elif source.ndim==5:order=(1,4,3,2,0);result=torch.channels_last_3d
    else:return torch.contiguous_format
    shape=source.shape;strides=source.stride()
    if strides[1]==0:return torch.contiguous_format
    minimum=0
    for dim in order:
        if shape[dim]==0 or strides[dim]<minimum:return torch.contiguous_format
        if dim==0 and minimum==strides[1]:return torch.contiguous_format
        minimum=strides[dim]*shape[dim]
    return result

def concatenate(tensors,dim=0,*,stack=False,sink=None,progress=None):
    if (type(tensors) is not list and type(tensors) is not tuple) or not tensors:
        raise k.StorageError("NONEMPTY_PLAIN_TENSOR_SEQUENCE_REQUIRED")
    for t in tensors:k.admit(t);require_canonical_bool(t)
    first=tensors[0]
    if any(t.dtype is not first.dtype for t in tensors):raise k.StorageError("CAT_DTYPE_PROMOTION_NOT_ADMITTED")
    if any(t.ndim!=first.ndim for t in tensors):raise k.StorageError("CAT_MIXED_RANK_NOT_ADMITTED")
    axis=dimension(dim,first.ndim,insert=stack)
    shape=list(first.shape)
    if stack:
        if any(tuple(t.shape)!=tuple(shape) for t in tensors):raise k.StorageError("STACK_SHAPE_MISMATCH")
        shape.insert(axis,len(tensors))
    else:
        if first.ndim==0:raise k.StorageError("CAT_SCALAR_NOT_ADMITTED")
        for t in tensors:
            if any(t.shape[j]!=shape[j] for j in range(first.ndim) if j!=axis):
                raise k.StorageError("CAT_SHAPE_MISMATCH")
        shape[axis]=sum(t.shape[axis] for t in tensors)
    formats=[suggested_format(t) for t in tensors]
    output_format=formats[0] if all(f is formats[0] for f in formats) else torch.contiguous_format
    if stack:
        if output_format is not torch.contiguous_format:raise k.StorageError('CHANNELS_LAST_STACK_NOT_YET_ADMITTED')
        output_format=torch.contiguous_format
    result=k.allocate(shape,dtype=first.dtype,memory_format=output_format)
    cursor=0
    for part,t in enumerate(tensors):
        start=cursor
        mapper=(lambda ix,part=part:ix[:axis]+(part,)+ix[axis:]) if stack else (
            lambda ix,start=start:ix[:axis]+(ix[axis]+start,)+ix[axis+1:])
        mapped_copy(t,result,mapper,sink=sink,progress=progress,operation="stack" if stack else "cat")
        if not stack:cursor+=t.shape[axis]
    return result

def flip(source,dims,*,sink=None,progress=None):
    k.admit(source);require_canonical_bool(source)
    if type(dims) is not list and type(dims) is not tuple:raise k.StorageError("PLAIN_DIMS_REQUIRED")
    axes=tuple(dimension(d,source.ndim) for d in dims)
    if len(set(axes))!=len(axes):raise k.StorageError("DUPLICATE_FLIP_DIMENSION")
    result=k.allocate(tuple(source.shape),dtype=source.dtype,like=source)
    mapped_copy(source,result,
        lambda ix:tuple(source.shape[j]-1-i if j in axes else i for j,i in enumerate(ix)),
        sink=sink,progress=progress,operation="flip")
    return result

def index_select(source,dim,index,*,sink=None,progress=None):
    k.admit(source);k.admit(index);require_canonical_bool(source)
    if source.ndim==0:raise k.StorageError("SCALAR_INDEX_SELECT_NOT_ADMITTED")
    axis=dimension(dim,source.ndim)
    if index.ndim>1 or not any(index.dtype is d for d in (torch.int32,torch.int64)):
        raise k.StorageError("INTEGER_SCALAR_OR_VECTOR_INDEX_REQUIRED")
    def index_value(j):
        return value_at(index,() if index.ndim==0 else (j,))
    # Validate every index before producing an output.
    for j in range(index.numel()):
        value=index_value(j)
        if not 0<=value<source.shape[axis]:raise k.StorageError("INDEX_SELECT_OUT_OF_RANGE")
    shape=list(source.shape);shape[axis]=index.numel()
    result=k.allocate(shape,dtype=source.dtype,memory_format=torch.contiguous_format)
    src,dst=k.raw(source),k.raw(result);width=source.element_size()
    for ordinal,ix in enumerate(k.coords(shape)):
        if ordinal%4096==0:k.tick(progress,operation="index_select",element=ordinal)
        source_ix=ix[:axis]+(index_value(ix[axis]),)+ix[axis+1:]
        si=k.offset(source,source_ix)*width;di=k.offset(result,ix)*width
        bits=bytes(src[si:si+width]);out=bytes(1 if b else 0 for b in bits) if source.dtype is torch.bool else bits
        k.emit(sink,"STORAGE_WRITE_INTENT",operation="index_select",src_byte=si,dst_byte=di,
               source_hex=bits.hex(),before_hex=None,after_hex=out.hex(),fresh_allocation=True)
        dst[di:di+width]=out
        k.emit(sink,"STORAGE_WRITE_DONE",operation="index_select",dst_byte=di,byte_count=width)
    return result

def scalar(source,truth=False,*,sink=None,progress=None):
    k.admit(source)
    if source.numel()!=1:raise k.StorageError("EXACTLY_ONE_ELEMENT_REQUIRED")
    ix=(0,)*source.ndim
    at=k.offset(source,ix)*source.element_size()
    bits=bytes(k.raw(source)[at:at+source.element_size()])
    value=decode_word(bits,source.dtype)
    k.emit(sink,"SCALAR_READ",operation="is_nonzero" if truth else "item",
           src_byte=at,source_hex=bits.hex(),dtype=str(source.dtype))
    return bool(value) if truth else value

def nonzero(source,*,sink=None,progress=None):
    k.admit(source);count=0
    for ordinal,ix in enumerate(k.coords(source.shape)):
        if ordinal%4096==0:k.tick(progress,operation="nonzero_count",element=ordinal)
        count+=value_at(source,ix)!=0
    shape=(count,source.ndim)
    # This is a declared native allocation, not native nonzero.
    with torch._C._DisableTorchDispatch(),torch.no_grad():
        result=torch.empty_strided(shape,(1,count),dtype=torch.int64,device="cpu")
    dst=k.raw(result);row=0
    for ordinal,ix in enumerate(k.coords(source.shape)):
        if ordinal%4096==0:k.tick(progress,operation="nonzero_write",element=ordinal)
        if value_at(source,ix)==0:continue
        for axis,i in enumerate(ix):
            at=(row+axis*count)*8;bits=int(i).to_bytes(8,"little",signed=True)
            k.emit(sink,"STORAGE_WRITE_INTENT",operation="nonzero",dst_byte=at,
                   before_hex=None,after_hex=bits.hex(),fresh_allocation=True)
            dst[at:at+8]=bits
            k.emit(sink,"STORAGE_WRITE_DONE",operation="nonzero",dst_byte=at,byte_count=8)
        row+=1
    return result

def execute(name,v,*,sink=None,progress=None):
    kw=dict(sink=sink,progress=progress)
    if name in ("aten.cat.default","aten.stack.default"):
        return concatenate(v["tensors"],v["dim"],stack=name=="aten.stack.default",**kw)
    if name=="aten.flip.default":return flip(v["self"],v["dims"],**kw)
    if name=="aten.index_select.default":return index_select(v["self"],v["dim"],v["index"],**kw)
    if name in ("aten._local_scalar_dense.default","aten.is_nonzero.default"):
        return scalar(v["self"],truth=name=="aten.is_nonzero.default",**kw)
    if name=="aten.nonzero.default":return nonzero(v["self"],**kw)
    raise k.StorageError("OPERATOR_NOT_IMPLEMENTED:"+name)
