"""Readable selection kernels. No native sort, topk, or gather call.

Sorting uses stable original-index tie breaking. For topk/unstable sort this is
an explicit diagnostic tie policy, NOT guaranteed identical to native ties.
"""
from __future__ import annotations
import math
from .core import Array, dim_axis, indices
from .training_ops import offset


def select(x, dim=-1, *, descending=False, k=None):
    if not x.shape:
        if dim not in (0,-1) or k not in (None,0,1):
            raise ValueError('SCALAR_SELECTION_ARGUMENT')
        return (x, Array((),(0,))) if k != 0 else (Array((0,),()),Array((0,),()))
    dim=dim_axis(dim,len(x.shape));n=x.shape[dim]
    count=n if k is None else k
    if type(count) is not int or not 0 <= count <= n:
        raise ValueError('TOPK_COUNT')
    shape=list(x.shape);shape[dim]=count;shape=tuple(shape)
    values=[0]*math.prod(shape);ranks=[0]*math.prod(shape)
    outer=x.shape[:dim]+x.shape[dim+1:]
    for ix in indices(outer):
        row=[x.at(ix[:dim]+(j,)+ix[dim:]) for j in range(n)]
        # NaN sorts beyond +inf, preserving NaN payload only in raw weight codec.
        def key(j):
            z=row[j];nan=isinstance(z,float) and math.isnan(z)
            return (1 if nan else 0,0 if nan else z)
        order=sorted(range(n),key=key,reverse=descending)[:count]
        for j,src in enumerate(order):
            at=offset(ix[:dim]+(j,)+ix[dim:],shape)
            values[at]=row[src];ranks[at]=src
    return Array(shape,tuple(values)),Array(shape,tuple(ranks))


def extremum(x, dim=None, *, keepdim=False, minimum=False, indices_only=False):
    if dim is None:
        if not x.data:raise ValueError('EMPTY_EXTREMUM')
        flat=Array((len(x.data),),x.data)
        v,i=extremum(flat,0,minimum=minimum)
        shape=(1,)*len(x.shape) if keepdim else ()
        return Array(shape,i.data) if indices_only else Array(shape,v.data)
    if not x.shape:
        if dim not in (0,-1):raise ValueError('DIM_OUT_OF_RANGE')
        pair=x,Array((),(0,))
        return pair[1] if indices_only else pair
    dim=dim_axis(dim,len(x.shape));n=x.shape[dim]
    if n==0:raise ValueError('EMPTY_EXTREMUM')
    outer=x.shape[:dim]+x.shape[dim+1:];vs=[];ids=[]
    for ix in indices(outer):
        row=[x.at(ix[:dim]+(j,)+ix[dim:]) for j in range(n)]
        chosen=0
        for j in range(1,n):
            a=row[chosen];z=row[j]
            if isinstance(a,float) and math.isnan(a):continue
            if (isinstance(z,float) and math.isnan(z)) or (z<a if minimum else z>a):chosen=j
        vs.append(row[chosen]);ids.append(chosen)
    shape=x.shape[:dim]+(1,)+x.shape[dim+1:] if keepdim else outer
    pair=Array(shape,tuple(vs)),Array(shape,tuple(ids))
    return pair[1] if indices_only else pair


def gather(x, dim, index):
    if len(x.shape)!=len(index.shape):raise ValueError('GATHER_RANK')
    dim=dim_axis(dim,len(x.shape))
    if any(index.shape[j]>x.shape[j] for j in range(len(x.shape)) if j!=dim):raise ValueError('GATHER_SHAPE')
    out=[]
    for ix,loc in zip(indices(index.shape),index.data):
        if type(loc) is not int or not 0<=loc<x.shape[dim]:raise IndexError('GATHER_INDEX')
        out.append(x.at(ix[:dim]+(loc,)+ix[dim+1:]))
    return Array(index.shape,tuple(out))
