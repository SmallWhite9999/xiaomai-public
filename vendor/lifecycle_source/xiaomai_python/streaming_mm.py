"""Tiled Python multiplication, not BLAS and not reduced-size weights.

Only storage/layout slicing and Tensor<->Python conversion remain native.
Each output accumulates k in exactly increasing order; dtype cast is at output,
as in the existing binary64 reference contract. Tile progress is resumable
observation, not an arithmetic checkpoint or a substitute for scalar trace.
"""
from __future__ import annotations
from . import core


def mm(a,b,*,record=None,progress=None,k_block=256,n_block=16):
    import torch
    if not any(type(a) is t for t in (torch.Tensor,torch.nn.Parameter)) or not any(type(b) is t for t in (torch.Tensor,torch.nn.Parameter)):
        raise ValueError('TENSOR_SUBCLASS_UNSUPPORTED')
    if a.device.type!='cpu' or b.device.type!='cpu' or a.layout!=torch.strided or b.layout!=torch.strided or a.is_complex() or b.is_complex():
        raise ValueError('STREAMING_CPU_REAL_STRIDED_ONLY')
    if len(a.shape)!=2 or len(b.shape)!=2 or a.shape[1]!=b.shape[0]:raise ValueError('MM_SHAPES')
    if min(k_block,n_block)<=0:raise ValueError('POSITIVE_TILE_REQUIRED')
    m,k=a.shape;_,n=b.shape
    values=[]
    for i in range(m):
        for j0 in range(0,n,n_block):
            width=min(n_block,n-j0);totals=[0]*width
            for k0 in range(0,k,k_block):
                end=min(k,k0+k_block)
                if progress:progress.work(operator='aten.mm.default',i=i,j_begin=j0,k_begin=k0,k_end=end,shape_a=list(a.shape),shape_b=list(b.shape))
                with torch.no_grad(),torch._C._DisableTorchDispatch():
                    left=a[i,k0:end].detach().tolist()
                    tile=b[k0:end,j0:j0+width].detach()
                    # Transpose only storage traversal. Each scalar product still
                    # accumulates k in increasing order, using Python arithmetic.
                    # Do not use sum(): CPython 3.12 float sum changes rounding.
                    right=tile.tolist() if record else tile.transpose(0,1).tolist()
                if record:
                    # Preserve the existing per-multiply trace order and values.
                    for h,x in enumerate(left):
                        row=right[h]
                        for j in range(width):
                            before=totals[j];totals[j]=before+x*row[j]
                            record('SCALAR_MULTIPLY_ADD',i=i,j=j0+j,k=k0+h,left=x,right=row[j],before=before,after=totals[j])
                else:
                    # A local accumulator avoids list reads/writes for every MAC.
                    # Only a bounded tile is materialized, even for real weights.
                    for j,column in enumerate(right):
                        total=totals[j]
                        for x,y in zip(left,column):
                            total=total+x*y
                        totals[j]=total
            values.extend(totals)
    return core.Array((m,n),tuple(values))


def bmm(a,b,*,record=None,progress=None):
    import torch
    if len(a.shape)!=3 or len(b.shape)!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=b.shape[1]:raise ValueError('BMM_SHAPES')
    output=[]
    for batch in range(a.shape[0]):
        with torch._C._DisableTorchDispatch():aa=a[batch];bb=b[batch]
        emit=(lambda kind,**fields:record(kind,batch=batch,**fields)) if record else None
        output.extend(mm(aa,bb,record=emit,progress=progress).data)
    return core.Array((a.shape[0],a.shape[1],b.shape[2]),tuple(output))
