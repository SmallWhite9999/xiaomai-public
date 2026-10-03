"""Python view geometry; native as_strided retains storage/version/autograd machinery.

This is PYTHON_VIEW_PLAN, not a claim that TensorImpl or autograd was rewritten.
No view/reshape/permute/slice/select/split kernel computes the geometry here.
"""
from __future__ import annotations
import math
import torch
from . import kernels as k

OPERATORS=frozenset("aten."+n for n in (
    "view.default","reshape.default","permute.default","transpose.int","t.default",
    "unsqueeze.default","squeeze.default","squeeze.dim","squeeze.dims","expand.default",
    "slice.Tensor","select.int","split.Tensor","split_with_sizes.default","unbind.int","alias.default"))

def integer_sequence(value):
    if type(value) is not list and type(value) is not tuple and type(value) is not torch.Size:
        raise k.StorageError("CONCRETE_INTEGER_SEQUENCE_REQUIRED")
    if any(type(v) is not int for v in value):raise k.StorageError("CONCRETE_INTEGER_SEQUENCE_REQUIRED")
    return tuple(value)

def axis(d,rank,insert=False,scalar=False):
    n=rank+1 if insert else max(rank,1) if scalar else rank
    if type(d) is not int or not -n<=d<n:raise k.StorageError("DIMENSION_OUT_OF_RANGE")
    return d%n

def new_view(t,shape,strides,off=None,*,sink=None,operation):
    off=t.storage_offset() if off is None else off
    k.emit(sink,"VIEW_PLAN",operation=operation,shape=list(shape),stride=list(strides),
           storage_offset=off,source_storage_address=t.untyped_storage().data_ptr(),
           native_constructor="aten.as_strided")
    with torch._C._DisableTorchDispatch():
        return torch.as_strided(t,shape,strides,off)

def inferred_shape(shape,numel):
    shape=integer_sequence(shape)
    if any(n<-1 for n in shape) or shape.count(-1)>1:raise k.StorageError("INVALID_VIEW_SHAPE")
    known=math.prod(n for n in shape if n!=-1)
    if -1 in shape:
        if known==0 or numel%known:raise k.StorageError("AMBIGUOUS_OR_INCOMPATIBLE_VIEW_SHAPE")
        shape=tuple(numel//known if n==-1 else n for n in shape)
    elif known!=numel:raise k.StorageError("VIEW_NUMEL_MISMATCH")
    return shape

def view_strides(oldshape,oldstrides,shape):
    if not oldshape:return k.contiguous_strides(shape)
    if math.prod(oldshape)==0:
        return tuple(oldstrides) if tuple(shape)==tuple(oldshape) else k.contiguous_strides(shape)
    result=[0]*len(shape)
    dst=len(shape)-1;source_span=1;target_span=1;base_stride=oldstrides[-1]
    # Split old metadata into contiguous chunks, then fit target dimensions.
    for src in range(len(oldshape)-1,-1,-1):
        source_span*=oldshape[src]
        boundary=(src==0 or (oldshape[src-1]!=1 and oldstrides[src-1]!=source_span*base_stride))
        if boundary:
            while dst>=0 and (target_span<source_span or shape[dst]==1):
                result[dst]=target_span*base_stride
                target_span*=shape[dst];dst-=1
            if target_span!=source_span:raise k.StorageError("VIEW_REQUIRES_COPY")
            if src>0:
                base_stride=oldstrides[src-1];source_span=1;target_span=1
    if dst!=-1:raise k.StorageError("VIEW_REQUIRES_COPY")
    return tuple(result)

def execute(name,v,*,sink=None,progress=None):
    t=k.admit(v["self"]);shape=tuple(t.shape);strides=tuple(t.stride());off=t.storage_offset();rank=t.ndim
    make=lambda sh,st,offset=None:new_view(t,sh,st,offset,sink=sink,operation=name)
    if name in ("aten.view.default","aten.reshape.default"):
        wanted=inferred_shape(v.get("size",v.get("shape")),t.numel())
        try:st=view_strides(shape,strides,wanted)
        except k.StorageError:
            if name!="aten.reshape.default":raise
            copied=k.clone(t,memory_format=torch.contiguous_format,sink=sink,progress=progress)
            return new_view(copied,wanted,k.contiguous_strides(wanted),0,sink=sink,operation=name)
        return make(wanted,st)
    if name=="aten.alias.default":return make(shape,strides)
    if name=="aten.permute.default":
        ds=integer_sequence(v["dims"])
        if len(ds)!=rank:raise k.StorageError("PERMUTE_RANK_MISMATCH")
        ds=tuple(axis(d,rank) for d in ds)
        if len(set(ds))!=rank:raise k.StorageError("DUPLICATE_PERMUTE_AXIS")
        return make(tuple(shape[d] for d in ds),tuple(strides[d] for d in ds))
    if name in ("aten.transpose.int","aten.t.default"):
        if name=="aten.t.default":
            if rank>2:raise k.StorageError("T_REQUIRES_RANK_LE_2")
            a,b=0,max(0,rank-1)
        else:a,b=axis(v["dim0"],rank,scalar=True),axis(v["dim1"],rank,scalar=True)
        if rank==0:return make(shape,strides)
        ds=list(range(rank));ds[a],ds[b]=ds[b],ds[a]
        return make(tuple(shape[d] for d in ds),tuple(strides[d] for d in ds))
    if name=="aten.unsqueeze.default":
        d=axis(v["dim"],rank,insert=True)
        step=1 if d==rank else shape[d]*strides[d]
        return make(shape[:d]+(1,)+shape[d:],strides[:d]+(step,)+strides[d:])
    if name.startswith("aten.squeeze."):
        if name=="aten.squeeze.default":ds=tuple(range(rank))
        else:
            ds=(v["dim"],) if name=="aten.squeeze.dim" else integer_sequence(v["dim"])
            ds=tuple(axis(d,rank,scalar=True) for d in ds)
            if len(set(ds))!=len(ds):raise k.StorageError("DUPLICATE_SQUEEZE_AXIS")
        return make(tuple(n for j,n in enumerate(shape) if n!=1 or j not in ds),
                    tuple(s for j,s in enumerate(strides) if shape[j]!=1 or j not in ds))
    if name=="aten.expand.default":
        wanted=integer_sequence(v["size"])
        if len(wanted)<rank:raise k.StorageError("EXPAND_RANK_TOO_SMALL")
        sh=[0]*len(wanted);st=[0]*len(wanted);lead=len(wanted)-rank
        for d in range(len(wanted)-1,-1,-1):
            n=shape[d-lead] if d>=lead else 1
            stride=strides[d-lead] if d>=lead else (sh[d+1]*st[d+1] if d+1<len(wanted) else 1)
            size=wanted[d]
            if size==-1:
                if d<lead:raise k.StorageError("EXPAND_NEW_AXIS_MINUS_ONE")
                size=n
            if size<0:raise k.StorageError("EXPAND_NEGATIVE_SIZE")
            if n!=size:
                if n!=1:raise k.StorageError("EXPAND_NON_SINGLETON")
                stride=0
            sh[d]=size;st[d]=stride
        return make(tuple(sh),tuple(st))
    if name=="aten.slice.Tensor":
        d=axis(v["dim"],rank);step=v["step"]
        if type(step) is not int or step<=0:raise k.StorageError("SLICE_POSITIVE_STEP_REQUIRED")
        start,end,step=slice(v["start"],v["end"],step).indices(shape[d])
        sh=list(shape);st=list(strides);sh[d]=len(range(start,end,step));st[d]*=step
        return make(tuple(sh),tuple(st),off+start*strides[d])
    if name=="aten.select.int":
        d=axis(v["dim"],rank);index=v["index"]
        if type(index) is not int or not -shape[d]<=index<shape[d]:raise k.StorageError("SELECT_INDEX")
        index%=shape[d]
        return make(shape[:d]+shape[d+1:],strides[:d]+strides[d+1:],off+index*strides[d])
    if name in ("aten.split.Tensor","aten.split_with_sizes.default","aten.unbind.int"):
        d=axis(v["dim"],rank);outputs=[]
        if name=="aten.unbind.int":
            for i in range(shape[d]):
                if progress is not None:progress(operation=name,index=i)
                outputs.append(make(shape[:d]+shape[d+1:],strides[:d]+strides[d+1:],off+i*strides[d]))
            return outputs
        if name=="aten.split.Tensor":
            size=v["split_size"]
            if type(size) is not int or size<0 or size==0 and shape[d]!=0:raise k.StorageError("INVALID_SPLIT_SIZE")
            sizes=(0,) if shape[d]==0 else tuple(min(size,shape[d]-a) for a in range(0,shape[d],size))
        else:
            sizes=integer_sequence(v["split_sizes"])
            if any(n<0 for n in sizes) or sum(sizes)!=shape[d]:raise k.StorageError("SPLIT_SIZES_MISMATCH")
        cursor=0
        for n in sizes:
            sh=list(shape);sh[d]=n
            outputs.append(make(tuple(sh),strides,off+cursor*strides[d]));cursor+=n
        return outputs
    raise k.StorageError("VIEW_OPERATOR_NOT_IMPLEMENTED")
