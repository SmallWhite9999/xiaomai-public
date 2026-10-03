"""Python finite L2/infinity clipping with observable element updates.

Same conceptual norm/clip rule as clip_grad_norm_, with explicit Python scaled
sum-of-squares and dtype rounding. Native reduction-order bit identity is not
claimed. A nonfinite input is recorded and rejected, never replaced by zero.
"""
from __future__ import annotations
import math
import torch
from .adam_optimizer import AdamError, values, tensor, tensor_fingerprint, clone_data, sha, rounded
from .backend import StorageBackend
from . import kernels as k


def l2(values_):
    scale, ssq = 0., 1.
    for value in values_:
        a = abs(value)
        if a != 0.:
            if scale < a:
                ssq = 1. + ssq * (scale / a)**2; scale = a
            else:
                ssq += (a / scale)**2
    return scale * math.sqrt(ssq) if scale else 0.


def clip_grad_norm_python(named_parameters, max_norm, *, norm_type=2., sink=None,
                          record_values=True, before_commit=None):
    if (type(max_norm) is not int and type(max_norm) is not float) or not math.isfinite(max_norm) or max_norm < 0:
        raise AdamError('CLIP_MAX_NORM_FINITE_NONNEGATIVE')
    if (type(norm_type) is not int and type(norm_type) is not float) or norm_type not in (2., math.inf):
        raise AdamError('CLIP_ONLY_L2_OR_POSITIVE_INFINITY')
    def emit(kind, **fields):
        if sink is not None: sink(kind, **fields)
    selected = list(named_parameters); seen=set(); stores=set(); staged=[]; norms=[]
    parameter_stores={p.untyped_storage()._cdata for _,p in selected if type(p) is torch.Tensor or type(p) is torch.nn.Parameter}
    dtype = torch.float32
    for name, p in selected:
        if type(name) is not str or name in seen:raise AdamError('CLIP_UNIQUE_NAMES')
        seen.add(name);k.admit(p)
        g=p.grad
        if g is None:continue
        k.admit(g)
        if g.requires_grad or g.is_inference() or (g.dtype is not torch.float32 and g.dtype is not torch.float64):
            raise AdamError('CLIP_FINITE_CPU_FP32_FP64_GRADIENT')
        if tuple(g.shape)!=tuple(p.shape):raise AdamError('CLIP_GRADIENT_SHAPE')
        store=g.untyped_storage()._cdata
        if store in stores:raise AdamError('CLIP_SHARED_GRADIENT_STORAGE')
        if store in parameter_stores:raise AdamError('CLIP_PARAMETER_GRADIENT_ALIAS')
        stores.add(store)
        fingerprint=tensor_fingerprint(g);data=values(g)
        if any(not math.isfinite(v) for v in data):
            emit('PY_CLIP_NONFINITE_INPUT',parameter=name,raw_bytes=k.logical_bytes(g).hex(),repaired=False)
            raise AdamError('CLIP_NONFINITE_GRADIENT_NOT_REPAIRED')
        n=l2(data) if norm_type==2 else max((abs(v) for v in data),default=0.)
        n=rounded(n,g.dtype);norms.append(n)
        if g.dtype is torch.float64:dtype=torch.float64
        staged.append(dict(name=name,parameter=p,gradient=g,fingerprint=fingerprint,data=data,old=clone_data(g)))
    total=rounded(l2(norms) if norm_type==2 else max(norms,default=0.),dtype)
    if not math.isfinite(total):
        emit('PY_CLIP_NONFINITE_NORM',norm=total.hex(),gradient_values_repaired=False)
        raise AdamError('CLIP_NONFINITE_NORM_NOT_REPAIRED')
    # The epsilon is a clipping policy parameter fixed to the native API default.
    denominator=rounded(total+1e-6,dtype)
    coefficient=min(1.,rounded(float(max_norm)/denominator,dtype))
    for row in staged:
        g=row['gradient'];c=rounded(coefficient,g.dtype)
        out=[rounded(v*c,g.dtype) for v in row['data']]
        row['target']=tensor(tuple(g.shape),out,g.dtype);row['target_sha']=sha(row['target'])
        if record_values:
            for j,(old,new) in enumerate(zip(row['data'],out)):
                emit('PY_CLIP_ELEMENT',parameter=row['name'],index=j,gradient_before=old.hex(),
                     clip_coefficient=c.hex(),gradient_after=new.hex(),dtype=str(g.dtype))
    emit('PY_CLIP_PREPARED',norm=total.hex(),coefficient=coefficient.hex(),norm_type=float(norm_type).hex(),
         parameters=len(staged),native_norm=False)
    if before_commit is not None:before_commit()
    for r in staged:
        if r['parameter'].grad is not r['gradient'] or tensor_fingerprint(r['gradient'])!=r['fingerprint']:
            raise AdamError('CLIP_SOURCE_CHANGED')
    committed=[]
    try:
        with torch.no_grad(),StorageBackend(sink=sink,record_values=record_values):
            for r in staged:
                if r['parameter'].grad is not r['gradient'] or tensor_fingerprint(r['gradient'])!=r['fingerprint']:
                    raise AdamError('CLIP_SOURCE_CHANGED_DURING_WRITE')
                committed.append(r);r['gradient'].copy_(r['target'])
        if any(r['parameter'].grad is not r['gradient'] or sha(r['gradient'])!=r['target_sha'] for r in staged):raise AdamError('CLIP_WRITE_READBACK')
        for r in staged:r['committed_fingerprint']=tensor_fingerprint(r['gradient'])
        emit('PY_CLIP_COMMITTED',parameters=len(staged),norm=total.hex())
        if any(r['parameter'].grad is not r['gradient'] or tensor_fingerprint(r['gradient'])!=r['committed_fingerprint'] for r in staged):raise AdamError('CLIP_OBSERVER_CHANGED_RESULT')
    except BaseException:
        recovered=False
        try:
            with torch.no_grad(),StorageBackend():
                for r in reversed(committed):
                    r['gradient'].copy_(r['old'])
                    r['parameter'].grad=r['gradient']
            recovered=all(r['parameter'].grad is r['gradient'] and sha(r['gradient'])==sha(r['old']) for r in committed)
        finally:
            try:emit('PY_CLIP_FAILED',gradient_values_restored=recovered,native_version_counters_restored=False)
            except BaseException:pass
        raise
    return total
