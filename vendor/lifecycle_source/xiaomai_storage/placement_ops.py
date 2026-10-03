"""Python repetition, advanced-index geometry and scatter writes.

All selected values/coordinates are computed here. Allocation and TensorImpl /
autograd scheduling remain native. Duplicate-write and accumulate semantics are
not guessed: conflicting assignment targets are rejected. Accumulate=True uses
explicit sequential addition for admitted integer/FP32/FP64 finite-input domains.
"""
from __future__ import annotations
import math
import torch
from . import kernels as k
from . import index_ops as ix
from .numeric_words import broadcast_index,decode,encode_scalar

OPERATORS=frozenset('aten.'+n for n in ('repeat.default','repeat_interleave.self_int',
    'scatter.src','scatter_.src','slice_scatter.default','select_scatter.default',
    'index.Tensor','index_put.default','index_put_.default'))

def ints(xs):
    if type(xs) not in (list,tuple,torch.Size) or any(type(x) is not int for x in xs):
        raise k.StorageError('CONCRETE_INTEGER_SEQUENCE_REQUIRED')
    return tuple(xs)

def axis(d,rank):
    if type(d) is not int or not -rank<=d<rank:raise k.StorageError('DIMENSION_OUT_OF_RANGE')
    return d%rank

def guard(t):
    k.admit(t);ix.require_canonical_bool(t)

def disjoint(a,b):
    x,y=k.extent(a);u,v=k.extent(b)
    if x<v and u<y:raise k.StorageError('PLACEMENT_SOURCE_DESTINATION_ALIAS')

def write(src,dst,src_ix,dst_ix,*,sink,operation,fresh,accumulate=False):
    sw,dw=src.element_size(),dst.element_size()
    if src.dtype is not dst.dtype:raise k.StorageError('PLACEMENT_DTYPE_MISMATCH')
    source,target=k.raw(src),k.raw(dst);a=k.offset(src,src_ix)*sw;b=k.offset(dst,dst_ix)*dw
    source_bits=bytes(source[a:a+sw]);bits=source_bits
    if accumulate:
        old=decode(bytes(target[b:b+dw]),dst.dtype);value=decode(source_bits,src.dtype)
        bits=encode_scalar(old+value,dst.dtype)
        k.emit(sink,'PYTHON_ACCUMULATE',operation=operation,before_hex=bytes(target[b:b+dw]).hex(),source_hex=source_bits.hex(),after_hex=bits.hex())
    k.emit(sink,'STORAGE_WRITE_INTENT',operation=operation,src_byte=a,dst_byte=b,
       before_hex=None if fresh else bytes(target[b:b+dw]).hex(),source_hex=source_bits.hex(),after_hex=bits.hex(),fresh_allocation=fresh)
    target[b:b+dw]=bits
    k.emit(sink,'STORAGE_WRITE_DONE',operation=operation,dst_byte=b,byte_count=dw)

def mapped(source,shape,mapper,*,sink,progress,operation):
    result=k.allocate(shape,dtype=source.dtype,memory_format=torch.contiguous_format)
    for n,target_ix in enumerate(k.coords(shape)):
        if n%4096==0:k.tick(progress,operation=operation,element=n)
        write(source,result,mapper(target_ix),target_ix,sink=sink,operation=operation,fresh=True)
    return result

def repeat(source,repeats,*,sink=None,progress=None):
    guard(source);repeats=ints(repeats)
    if len(repeats)<source.ndim or any(n<0 for n in repeats):raise k.StorageError('INVALID_REPEAT')
    lead=len(repeats)-source.ndim;expanded=(1,)*lead+tuple(source.shape)
    shape=tuple(a*b for a,b in zip(expanded,repeats))
    mapper=lambda coord:tuple(coord[lead+j]%n for j,n in enumerate(source.shape))
    return mapped(source,shape,mapper,sink=sink,progress=progress,operation='repeat')

def repeat_interleave(source,repeats,dim=None,output_size=None,*,sink=None,progress=None):
    guard(source)
    if type(repeats) is not int or repeats<0:raise k.StorageError('INVALID_REPEAT_COUNT')
    if dim is None:
        shape=(source.numel()*repeats,)
        def mapper(coord):
            flat=coord[0]//repeats;result=[0]*source.ndim
            for j in range(source.ndim-1,-1,-1):result[j]=flat%source.shape[j];flat//=source.shape[j]
            return tuple(result)
    else:
        if source.ndim==0:raise k.StorageError('SCALAR_REPEAT_INTERLEAVE_DIM_NOT_ADMITTED')
        d=axis(dim,source.ndim);shape=list(source.shape);shape[d]*=repeats;shape=tuple(shape)
        mapper=lambda coord:coord[:d]+(coord[d]//repeats,)+coord[d+1:]
    expected=shape[0] if dim is None else shape[d]
    if output_size is not None and output_size!=expected:raise k.StorageError('REPEAT_OUTPUT_SIZE_MISMATCH')
    return mapped(source,shape,mapper,sink=sink,progress=progress,operation='repeat_interleave')

def scatter(base,dim,index,source,*,inplace=False,sink=None,progress=None):
    guard(base);guard(source);k.admit(index);d=axis(dim,base.ndim)
    if base.dtype is not source.dtype:raise k.StorageError('SCATTER_DTYPE_MISMATCH')
    if index.dtype is not torch.int64 or index.ndim!=base.ndim or source.ndim!=base.ndim:
        raise k.StorageError('SCATTER_RANK_OR_INDEX_DTYPE')
    if any(index.shape[j]>source.shape[j] or j!=d and index.shape[j]>base.shape[j] for j in range(base.ndim)):
        raise k.StorageError('SCATTER_SHAPE')
    if not k.nonoverlap(base):raise k.StorageError('SCATTER_OVERLAPPING_BASE')
    if inplace:disjoint(base,source);disjoint(base,index)
    seen=set()
    def destination(coord):
        i=ix.value_at(index,coord)
        if not 0<=i<base.shape[d]:raise k.StorageError('SCATTER_INDEX_RANGE')
        return coord[:d]+(i,)+coord[d+1:]
    for n,coord in enumerate(k.coords(index.shape)):
        if n%4096==0:k.tick(progress,operation='scatter',stage='validate',element=n)
        dest=destination(coord)
        if dest in seen:raise k.StorageError('DUPLICATE_SCATTER_TARGET_NOT_ADMITTED')
        seen.add(dest)
    result=base if inplace else k.clone(base,sink=sink,progress=progress)
    for n,coord in enumerate(k.coords(index.shape)):
        if n%4096==0:k.tick(progress,operation='scatter',stage='write',element=n)
        write(source,result,coord,destination(coord),sink=sink,operation='scatter',fresh=False)
    return result


def clone_preserving_storage(source,*,sink=None,progress=None):
    """The scatter clone preserves stride, offset and the underlying storage span."""
    k.admit(source)
    count=source.untyped_storage().nbytes()//source.element_size()
    backing=k.allocate((count,),dtype=source.dtype,memory_format=torch.contiguous_format)
    src,dst=k.raw(source),k.raw(backing)
    if source.dtype is torch.bool and any(b not in (0,1) for b in src):
        raise k.StorageError('NONCANONICAL_BOOL_SCATTER_STORAGE_NOT_ADMITTED')
    for at in range(0,len(src),k.BLOCK_BYTES):
        data=bytes(src[at:at+k.BLOCK_BYTES])
        k.tick(progress,operation='clone_storage_span',byte_offset=at,byte_count=len(data))
        k.emit(sink,'STORAGE_WRITE_INTENT',operation='clone_storage_span',src_byte=at,dst_byte=at,
               before_hex=None,source_hex=data.hex(),after_hex=data.hex(),fresh_allocation=True)
        dst[at:at+len(data)]=data
        k.emit(sink,'STORAGE_WRITE_DONE',operation='clone_storage_span',dst_byte=at,byte_count=len(data))
    with torch._C._DisableTorchDispatch():
        return torch.as_strided(backing,source.shape,source.stride(),source.storage_offset())

def selective_scatter(base,source,dim,*,selected=None,start=None,end=None,step=1,sink=None,progress=None):
    guard(base);guard(source);d=axis(dim,base.ndim)
    if base.dtype is not source.dtype:raise k.StorageError('SCATTER_DTYPE_MISMATCH')
    if not k.nonoverlap(base):raise k.StorageError('SCATTER_OVERLAPPING_BASE')
    if selected is not None:
        if type(selected) is not int or not -base.shape[d]<=selected<base.shape[d]:raise k.StorageError('SELECT_SCATTER_INDEX')
        selected%=base.shape[d];expected=tuple(base.shape[:d])+tuple(base.shape[d+1:])
        mapping=lambda coord:coord[:d]+(selected,)+coord[d:]
        operation='select_scatter'
    else:
        if type(step) is not int or step<=0:raise k.StorageError('SLICE_SCATTER_STEP')
        start,end,step=slice(start,end,step).indices(base.shape[d]);expected=list(base.shape);expected[d]=len(range(start,end,step));expected=tuple(expected)
        mapping=lambda coord:coord[:d]+(start+coord[d]*step,)+coord[d+1:]
        operation='slice_scatter'
    if tuple(source.shape)!=expected:raise k.StorageError('SELECTIVE_SCATTER_SOURCE_SHAPE')
    result=clone_preserving_storage(base,sink=sink,progress=progress)
    for n,coord in enumerate(k.coords(source.shape)):
        if n%4096==0:k.tick(progress,operation=operation,element=n)
        write(source,result,coord,mapping(coord),sink=sink,operation=operation,fresh=False)
    return result

def common_shape(shapes):
    rank=max(map(len,shapes),default=0);out=[1]*rank
    for shape in shapes:
        for j,n in enumerate((1,)*(rank-len(shape))+shape):
            if out[j]!=n and out[j]!=1 and n!=1:raise k.StorageError('ADVANCED_INDEX_BROADCAST')
            if out[j]==1:out[j]=n
    return tuple(out)

class IndexPlan:
    """Concrete integer tensors plus one-dimensional boolean masks; None means full axis."""
    def __init__(self,source,indices,progress=None):
        guard(source)
        if type(indices) not in (tuple,list) or not indices or len(indices)>source.ndim:
            raise k.StorageError('ADVANCED_INDEX_LIST_RANK')
        self.source=source;self.entries={};shapes=[]
        for d,index in enumerate(indices):
            if index is None:continue
            k.admit(index)
            if index.dtype is torch.bool:
                if index.ndim!=1 or index.shape[0]!=source.shape[d]:raise k.StorageError('ONLY_AXIS_BOOLEAN_MASK_ADMITTED')
                positions=tuple(i for i in range(index.numel()) if ix.value_at(index,(i,)))
                shape=(len(positions),)
                getter=lambda coord,positions=positions:positions[coord[0]]
            else:
                if not any(index.dtype is dt for dt in (torch.int32,torch.int64)):raise k.StorageError('INTEGER_OR_BOOL_INDEX_REQUIRED')
                shape=tuple(index.shape);getter=lambda coord,index=index:ix.value_at(index,coord)
            self.entries[d]=(shape,getter);shapes.append(shape)
        if not self.entries:raise k.StorageError('AT_LEAST_ONE_ADVANCED_INDEX_REQUIRED')
        self.adv_shape=common_shape(shapes);axes=sorted(self.entries);self.basic=[d for d in range(source.ndim) if d not in self.entries]
        self.contiguous=axes==list(range(axes[0],axes[-1]+1))
        self.before=axes[0] if self.contiguous else 0
        if self.contiguous:
            self.output_shape=tuple(source.shape[:axes[0]])+self.adv_shape+tuple(source.shape[axes[-1]+1:])
        else:self.output_shape=self.adv_shape+tuple(source.shape[d] for d in self.basic)
        for n,coord in enumerate(k.coords(self.adv_shape)):
            if n%4096==0:k.tick(progress,operation='advanced_index',stage='validate',element=n)
            self.advanced_values(coord)
    def advanced_values(self,coord):
        out={}
        for d,(shape,getter) in self.entries.items():
            sub=broadcast_index(shape,self.adv_shape)(coord);value=getter(sub)
            if not -self.source.shape[d]<=value<self.source.shape[d]:raise k.StorageError('ADVANCED_INDEX_OUT_OF_RANGE')
            out[d]=value%self.source.shape[d]
        return out
    def target(self,coord):
        n=len(self.adv_shape);off=self.before
        advanced=self.advanced_values(coord[off:off+n])
        basic=coord[:off]+coord[off+n:];advanced.update(zip(self.basic,basic))
        return tuple(advanced[d] for d in range(self.source.ndim))

def advanced_index(source,indices,*,sink=None,progress=None):
    plan=IndexPlan(source,indices,progress)
    return mapped(source,plan.output_shape,plan.target,sink=sink,progress=progress,operation='advanced_index')

def index_put(base,indices,values,*,accumulate=False,inplace=False,sink=None,progress=None):
    if accumulate and base.dtype not in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64,torch.float32,torch.float64):
        raise k.StorageError('ACCUMULATION_DTYPE_NOT_ADMITTED')
    guard(base);guard(values)
    if base.dtype is not values.dtype:raise k.StorageError('INDEX_PUT_DTYPE')
    if not k.nonoverlap(base):raise k.StorageError('INDEX_PUT_OVERLAPPING_BASE')
    if inplace:
        disjoint(base,values)
        for index in indices:
            if index is not None:disjoint(base,index)
    plan=IndexPlan(base,indices,progress);getvalue=broadcast_index(tuple(values.shape),plan.output_shape)
    seen=set()
    for n,coord in enumerate(k.coords(plan.output_shape)):
        if n%4096==0:k.tick(progress,operation='index_put',stage='uniqueness',element=n)
        dest=plan.target(coord)
        if dest in seen and not accumulate:raise k.StorageError('DUPLICATE_INDEX_PUT_TARGET_NOT_ADMITTED')
        if accumulate:
            a=ix.value_at(base,dest);b=ix.value_at(values,getvalue(coord))
            if any(type(x) is float and not math.isfinite(x) for x in (a,b)):
                k.emit(sink,'ACCUMULATION_REJECTED',operation='index_put',reason='nonfinite_input')
                raise k.StorageError('NONFINITE_ACCUMULATION_NOT_ADMITTED')
        seen.add(dest)
    result=base if inplace else k.clone(base,sink=sink,progress=progress)
    for n,coord in enumerate(k.coords(plan.output_shape)):
        if n%4096==0:k.tick(progress,operation='index_put',stage='write',element=n)
        write(values,result,getvalue(coord),plan.target(coord),sink=sink,operation='index_put',fresh=False,accumulate=accumulate)
    return result

def execute(name,v,*,sink=None,progress=None):
    options=dict(sink=sink,progress=progress)
    if name=='aten.repeat.default':return repeat(v['self'],v['repeats'],**options)
    if name=='aten.repeat_interleave.self_int':return repeat_interleave(v['self'],v['repeats'],v['dim'],v['output_size'],**options)
    if name in ('aten.scatter.src','aten.scatter_.src'):return scatter(v['self'],v['dim'],v['index'],v['src'],inplace=name=='aten.scatter_.src',**options)
    if name=='aten.slice_scatter.default':return selective_scatter(v['self'],v['src'],v['dim'],start=v['start'],end=v['end'],step=v['step'],**options)
    if name=='aten.select_scatter.default':return selective_scatter(v['self'],v['src'],v['dim'],selected=v['index'],**options)
    if name=='aten.index.Tensor':return advanced_index(v['self'],v['indices'],**options)
    if name in ('aten.index_put.default','aten.index_put_.default'):return index_put(v['self'],v['indices'],v['values'],accumulate=v['accumulate'],inplace=name=='aten.index_put_.default',**options)
    raise k.StorageError('PLACEMENT_OPERATOR_NOT_IMPLEMENTED:'+name)
