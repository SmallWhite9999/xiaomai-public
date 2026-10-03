"""Python scalar training/attention mathematics. No torch or numpy arithmetic.

Python binary64 sequential arithmetic, with cast handled at the dispatch port.
This is a diagnostic implementation, not a Tensor Core bit-exact emulator.
"""
from __future__ import annotations
import math
from .core import Array, indices, scalar, elementwise, reduce, divide, log, exp, softmax, dim_axis, power as real_power


def offset(ix, shape):
    value = 0
    for i, n in zip(ix, shape):
        value = value*n+i
    return value


def reduced(a: Array, reduction: int):
    if reduction == 0:
        return a
    if reduction not in (1, 2):
        raise ValueError('INVALID_LOSS_REDUCTION')
    total = sum(a.data)
    return scalar(divide(total, len(a.data)) if reduction == 1 else total)


def nll_layout(x, target, weight):
    if len(x.shape) < 1:
        raise ValueError('NLL_RANK')
    classes = x.shape[0] if len(x.shape) == 1 else x.shape[1]
    shape = () if len(x.shape) == 1 else (x.shape[0],)+x.shape[2:]
    if target.shape != shape or (weight is not None and weight.shape != (classes,)):
        raise ValueError('NLL_TARGET_OR_WEIGHT_SHAPE')
    return classes


def nll_forward(x, target, weight, reduction, ignore_index):
    classes = nll_layout(x, target, weight)
    values = []; total_weight = 0.0
    for ix, ident in zip(indices(target.shape), target.data):
        if type(ident) is not int:
            raise TypeError('NLL_TARGET_INTEGER_REQUIRED')
        if ident == ignore_index:
            values.append(0.0); continue
        if not 0 <= ident < classes:
            raise IndexError('NLL_TARGET_OUT_OF_BOUNDS')
        w = weight.data[ident] if weight is not None else 1.0
        at = (ident,) if len(x.shape) == 1 else (ix[0], ident)+ix[1:]
        values.append(-x.at(at)*w); total_weight += w
    if reduction == 0:
        return Array(target.shape, tuple(values)), scalar(total_weight if len(x.shape) == 1 else 0.0)
    if reduction not in (1, 2):
        raise ValueError('INVALID_LOSS_REDUCTION')
    total = sum(values)
    return scalar(divide(total, total_weight) if reduction == 1 else total), scalar(total_weight)


def nll_backward(grad, x, target, weight, reduction, ignore_index, total_weight):
    classes = nll_layout(x, target, weight)
    if (reduction == 0 and grad.shape != target.shape) or (reduction != 0 and len(grad.data) != 1):
        raise ValueError('NLL_GRAD_SHAPE')
    out = [0.0]*len(x.data)
    for ordinal, (ix, ident) in enumerate(zip(indices(target.shape), target.data)):
        if ident == ignore_index:
            continue
        if type(ident) is not int or not 0 <= ident < classes:
            raise IndexError('NLL_TARGET_OUT_OF_BOUNDS')
        value = grad.data[ordinal if reduction == 0 else 0]
        if reduction == 1:
            value = divide(value, total_weight.data[0])
        at = (ident,) if len(x.shape) == 1 else (ix[0], ident)+ix[1:]
        out[offset(at, x.shape)] = -value*(weight.data[ident] if weight is not None else 1.0)
    return Array(x.shape, tuple(out))


def regression_loss(x, target, reduction, *, beta=None, grad=None):
    if x.shape != target.shape:
        raise ValueError('REGRESSION_TARGET_SHAPE')
    if beta is not None and not beta >= 0:
        raise ValueError('NEGATIVE_SMOOTH_L1_BETA')
    values = []
    for a, b in zip(x.data, target.data):
        z = a-b
        if grad is None:
            value = z*z if beta is None else (.5*z*z/beta if beta != 0 and abs(z) < beta else abs(z)-.5*beta)
        else:
            # Match the low-level ATen kernel's comparison order. NaN must
            # reach arithmetic, not disappear through a boolean sign formula.
            # beta=0 in public F.smooth_l1_loss is a separate L1 decomposition.
            if beta is None:
                value = 2*z
            elif z <= -beta:
                value = -1.0
            elif z >= beta:
                value = 1.0
            else:
                value = divide(z, beta)
        values.append(value)
    a = Array(x.shape, tuple(values))
    if grad is None:
        return reduced(a, reduction)
    scale = divide(1, len(values)) if reduction == 1 else 1.0
    if reduction not in (0,1,2):
        raise ValueError('INVALID_LOSS_REDUCTION')
    return elementwise(lambda g, v: g*v*scale, grad, a)


def vector_norm(x, order=2, dims=None, keepdim=False):
    dims = tuple(range(len(x.shape))) if dims is None else tuple(dims)
    if not dims:
        dims = tuple(range(len(x.shape)))
    axes = tuple(dim_axis(d, len(x.shape)) for d in dims)
    if len(set(axes)) != len(axes):
        raise ValueError('DUPLICATE_NORM_DIMENSION')
    if order == 0:
        return reduce(elementwise(lambda z: float(z != 0), x), axes, keepdim=keepdim)
    absolute = elementwise(abs, x)
    reduced_size=math.prod(x.shape[d] for d in axes)
    if reduced_size==0 and order<0:raise ValueError('EMPTY_NEGATIVE_ORDER_NORM')
    if reduced_size==1:
        shape=tuple(1 if i in axes else n for i,n in enumerate(x.shape)) if keepdim else tuple(n for i,n in enumerate(x.shape) if i not in axes)
        return Array(shape,absolute.data)
    if order == math.inf:
        return reduce(absolute, axes, maximum=True, keepdim=keepdim)
    if order == -math.inf:
        return elementwise(lambda z:-z, reduce(elementwise(lambda z:-z, absolute), axes, maximum=True, keepdim=keepdim))
    if not math.isfinite(order):
        raise ValueError('INVALID_NORM_ORDER')
    def power(z):
        if z == 0 and order < 0:
            return math.inf
        return real_power(z,order)
    accum = reduce(elementwise(power, absolute), axes, keepdim=keepdim)
    return elementwise(lambda z: real_power(z,1/order) if z != 0 or order > 0 else math.inf, accum)


def attention_forward(q, k, v, *, causal=False, mask=None, scale=None):
    """Dense CPU SDPA contract: B,H,L,D, dropout=0, explicit additive/bool mask.

    The output and logsumexp are computed in Python. Softmax is zero on fully
    masked rows, and the saved LSE is +0 for those rows (CPU kernel contract).
    No cache, random dropout, GQA repetition, or packed layout is fabricated.
    """
    if any(len(a.shape) != 4 for a in (q,k,v)):
        raise ValueError('ATTENTION_REQUIRES_4D')
    b, h, l, d = q.shape; bk, hk, s, dk = k.shape
    if (bk,hk,dk) != (b,h,d) or v.shape[:3] != (b,h,s) or d <= 0:
        raise ValueError('ATTENTION_SHAPES')
    dv = v.shape[-1]; scale = 1/math.sqrt(d) if scale is None else scale
    scores = []; probabilities = []; lses = []; out = []
    for bi in range(b):
        for hi in range(h):
            for i in range(l):
                row = []
                for j in range(s):
                    score = sum(q.at((bi,hi,i,t))*k.at((bi,hi,j,t)) for t in range(d))*scale
                    if causal and j > i:
                        score = -math.inf
                    if mask is not None:
                        from .core import bat
                        m = bat(mask, (bi,hi,i,j))
                        score = (score if m else -math.inf) if type(m) is bool else score+m
                    row.append(score)
                pr = softmax(Array((s,), tuple(row)), 0, safe_all_masked=True).data if s else ()
                if not row or all(z == -math.inf for z in row):
                    lse = 0.0
                elif any(math.isnan(z) or z == math.inf for z in row):
                    lse = math.nan
                else:
                    high=max(row);lse=high+math.log(sum(math.exp(z-high) for z in row))
                scores.extend(row);probabilities.extend(pr);lses.append(lse)
                for t in range(dv):
                    out.append(sum(pr[j]*v.at((bi,hi,j,t)) for j in range(s)))
    return Array((b,h,l,dv),tuple(out)), Array((b,h,l),tuple(lses)), Array((b,h,l,s),tuple(probabilities))


def attention_backward(grad, q, k, v, *, causal=False, mask=None, scale=None):
    _, _, p = attention_forward(q,k,v,causal=causal,mask=mask,scale=scale)
    b,h,l,d=q.shape;s=k.shape[2];dv=v.shape[-1]
    if grad.shape != (b,h,l,dv):
        raise ValueError('ATTENTION_GRAD_SHAPE')
    scale=1/math.sqrt(d) if scale is None else scale
    dq=[0.0]*len(q.data);dk=[0.0]*len(k.data);dval=[0.0]*len(v.data)
    for bi in range(b):
        for hi in range(h):
            for i in range(l):
                dp=[sum(grad.at((bi,hi,i,t))*v.at((bi,hi,j,t)) for t in range(dv)) for j in range(s)]
                mean=sum(p.at((bi,hi,i,j))*dp[j] for j in range(s))
                for j in range(s):
                    probability=p.at((bi,hi,i,j));ds=probability*(dp[j]-mean)*scale
                    for t in range(d):
                        dq[offset((bi,hi,i,t),q.shape)]+=ds*k.at((bi,hi,j,t))
                        dk[offset((bi,hi,j,t),k.shape)]+=ds*q.at((bi,hi,i,t))
                    for t in range(dv):
                        dval[offset((bi,hi,j,t),v.shape)]+=probability*grad.at((bi,hi,i,t))
    return Array(q.shape,tuple(dq)),Array(k.shape,tuple(dk)),Array(v.shape,tuple(dval))


def compose_attention_mask(query_length, key_length, *, mask=None, causal=False, query_offset=0):
    """Pure-Python explicit mask; use it with native is_causal=False.

    Native versions disagree on accepting causal+mask simultaneously. Composition
    is declared, never a silent fallback after an exception. query_offset is
    explicit (0 means native upper-left causality, not cached-decode offset).
    """
    if any(type(n) is not int or n < 0 for n in (query_length,key_length,query_offset)):
        raise ValueError('ATTENTION_MASK_DIMENSIONS')
    from .core import broadcast_shape, bat
    shape=(query_length,key_length) if mask is None else broadcast_shape(mask.shape,(query_length,key_length))
    if shape[-2:] != (query_length,key_length):
        raise ValueError('ATTENTION_MASK_BROADCAST')
    boolean=mask is None or all(type(x) is bool for x in mask.data)
    values=[]
    for ix in indices(shape):
        valid=not causal or ix[-1] <= ix[-2]+query_offset
        if boolean:
            values.append(valid and (True if mask is None else bat(mask,ix)))
        else:
            values.append((0.0 if valid else -math.inf)+bat(mask,ix))
    return Array(shape,tuple(values))
