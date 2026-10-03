"""Scalar Python arithmetic. No torch/numpy matmul, reductions or convolutions.

Declared semantics: Python binary64 arithmetic with sequential reduction; libm
transcendentals; conversion to the output tensor's dtype happens at dispatch.
This is a readable reference, not a BF16 Tensor Core bitwise emulator.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product, repeat
import math
from typing import Callable

@dataclass(frozen=True)
class Array:
    shape: tuple[int, ...]
    data: tuple
    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in self.shape):
            raise ValueError('INVALID_SHAPE')
        if math.prod(self.shape) != len(self.data):
            raise ValueError('SHAPE_DATA_MISMATCH')
    def at(self, index):
        if len(index) != len(self.shape): raise IndexError('RANK_MISMATCH')
        pos = 0
        for i, n in zip(index, self.shape):
            if not 0 <= i < n: raise IndexError('INDEX_BOUNDS')
            pos = pos * n + i
        return self.data[pos]
    def reshape(self, shape): return Array(tuple(shape), self.data)

def indices(shape):
    return product(*(range(n) for n in shape))

def broadcast_shape(*shapes):
    ndim = max(map(len, shapes), default=0)
    out = [1] * ndim
    for shape in shapes:
        for i, n in enumerate((1,) * (ndim-len(shape)) + tuple(shape)):
            if out[i] != n and out[i] != 1 and n != 1:
                raise ValueError('BROADCAST_MISMATCH')
            if out[i] == 1: out[i] = n
    return tuple(out)

def bat(a: Array, index):
    part = index[len(index)-len(a.shape):] if a.shape else ()
    return a.at(tuple(0 if n == 1 else i for n, i in zip(a.shape, part)))

def scalar(x): return Array((), (x,))

def _broadcast_values(a, shape):
    """Visit logical values in row-major order without per-value coordinate dicts.

    This is Python indexing only. Contiguous tails are emitted as bounded tuple
    slices; broadcast axes repeat their source slice, never materialize an
    expanded tensor. No arithmetic/reduction is delegated to a native library.
    """
    if a.shape == shape:
        yield from a.data
        return
    size=math.prod(shape)
    if size == 0:return
    if len(a.data) == 1:
        yield from repeat(a.data[0],size)
        return
    if len(shape)>128:
        # Array is not limited to torch's practical ranks. Keep the original
        # iterative coordinate path for unusual ranks instead of recursing.
        for index in indices(shape):yield bat(a,index)
        return
    padded=(1,)*(len(shape)-len(a.shape))+a.shape
    strides=[];stride=1
    for n in reversed(padded):
        strides.append(stride);stride*=n
    strides.reverse()
    tail=len(shape)
    while tail and padded[tail-1]==shape[tail-1]:tail-=1
    block=math.prod(shape[tail:])
    def visit(dim,offset):
        if block==1 and dim==tail-1 and padded[dim]==1:
            yield from repeat(a.data[offset],shape[dim])
            return
        if dim==tail:
            # Bound transient slicing even when a matching tail is very large.
            for start in range(offset,offset+block,65536):
                yield from a.data[start:min(offset+block,start+65536)]
            return
        step=0 if padded[dim]==1 else strides[dim]
        for i in range(shape[dim]):yield from visit(dim+1,offset+i*step)
    yield from visit(0,0)

def elementwise(fn: Callable, *arrays: Array):
    shape = broadcast_shape(*(a.shape for a in arrays))
    if not arrays:return Array(shape,(fn(),))
    return Array(shape, tuple(map(fn,*(_broadcast_values(a,shape) for a in arrays))))

def dim_axis(dim, rank):
    if not -rank <= dim < rank: raise ValueError('DIM_OUT_OF_RANGE')
    return dim % rank

def reduce(a: Array, dims, *, mean=False, keepdim=False, maximum=False):
    if dims is None or len(dims) == 0: dims = tuple(range(len(a.shape)))
    dims = tuple(dim_axis(d, len(a.shape)) for d in dims)
    if len(set(dims)) != len(dims): raise ValueError('DUPLICATE_REDUCTION_DIM')
    selected = set(dims)
    out_shape = tuple(1 if j in selected else n for j,n in enumerate(a.shape)) if keepdim else tuple(n for j,n in enumerate(a.shape) if j not in selected)
    red_shape = tuple(a.shape[j] for j in dims)
    values=[]
    # Common last-axis reductions have a contiguous traversal. Preserve the
    # original dims order, accumulator initialization and per-element additions;
    # neither sum() nor a reassociated/compensated native reduction is used.
    if dims == tuple(range(len(a.shape)-len(dims),len(a.shape))):
        count=math.prod(red_shape)
        for row in range(math.prod(out_shape)):
            result=-math.inf if maximum else 0
            for i in range(row*count,(row+1)*count):
                value=a.data[i]
                if maximum:
                    result=math.nan if ((isinstance(value,float) and math.isnan(value)) or (isinstance(result,float) and math.isnan(result))) else max(result,value)
                else:result+=value
            if maximum and count==0:raise ValueError('EMPTY_MAX_REDUCTION')
            if mean:result=result/count if count else math.nan
            values.append(result)
        return Array(out_shape,tuple(values))
    for out in indices(out_shape):
        fixed={}; offset=0
        for j in range(len(a.shape)):
            if j not in selected:
                fixed[j]=out[j] if keepdim else out[offset]; offset+=1
        result = -math.inf if maximum else 0
        count = 0
        for red in indices(red_shape):
            at=dict(fixed); at.update(zip(dims,red))
            value=a.at(tuple(at[j] for j in range(len(a.shape))))
            if maximum:
                result=math.nan if ((isinstance(value,float) and math.isnan(value)) or (isinstance(result,float) and math.isnan(result))) else max(result,value)
            else: result += value
            count += 1
        if maximum and count==0: raise ValueError('EMPTY_MAX_REDUCTION')
        if mean: result = result/count if count else math.nan
        values.append(result)
    return Array(out_shape,tuple(values))

def mm(a: Array,b: Array,*,record=None):
    if len(a.shape)!=2 or len(b.shape)!=2 or a.shape[1]!=b.shape[0]: raise ValueError('MM_SHAPES')
    m,k=a.shape;_,n=b.shape; output=[]
    for i in range(m):
        for j in range(n):
            total=0
            for h in range(k):
                left=a.at((i,h));right=b.at((h,j));before=total
                total += left*right
                if record is not None:
                    record('SCALAR_MULTIPLY_ADD',i=i,j=j,k=h,left=left,right=right,before=before,after=total)
            output.append(total)
    return Array((m,n),tuple(output))

def bmm(a,b,*,record=None):
    if len(a.shape)!=3 or len(b.shape)!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=b.shape[1]: raise ValueError('BMM_SHAPES')
    B,M,K=a.shape;_,_,N=b.shape;output=[]
    for batch in range(B):
        x=Array((M,K),a.data[batch*M*K:(batch+1)*M*K]);y=Array((K,N),b.data[batch*K*N:(batch+1)*K*N])
        callback=(lambda event,**fields:record(event,batch=batch,**fields)) if record is not None else None
        output.extend(mm(x,y,record=callback).data)
    return Array((B,M,N),tuple(output))

def softmax(a, dim, *, log=False, safe_all_masked=False):
    dim=dim_axis(dim,len(a.shape));N=a.shape[dim]
    if N==0:return a
    values=[0.0]*len(a.data)
    outer=a.shape[:dim]+a.shape[dim+1:]
    for ix in indices(outer):
        row=[float(a.at(ix[:dim]+(j,)+ix[dim:])) for j in range(N)]
        if any(math.isnan(x) or x==math.inf for x in row): r=[math.nan]*N
        elif all(x==-math.inf for x in row): r=[0.0 if safe_all_masked else math.nan]*N
        else:
            m=max(row);exps=[math.exp(x-m) for x in row];den=sum(exps)
            r=[(x-m)-math.log(den) for x in row] if log else [x/den for x in exps]
        for j,value in enumerate(r):
            pos=0
            for index,n in zip(ix[:dim]+(j,)+ix[dim:],a.shape):pos=pos*n+index
            values[pos]=value
    return Array(a.shape,tuple(values))

def embedding(weight,ids):
    if len(weight.shape)!=2:raise ValueError('EMBEDDING_WEIGHT_RANK')
    N,D=weight.shape;out=[]
    for i in ids.data:
        if type(i) is not int or not 0<=i<N:raise IndexError('EMBEDDING_INDEX')
        out.extend(weight.data[i*D:(i+1)*D])
    return Array(ids.shape+(D,),tuple(out))

def convolution(x,w,bias,*,stride,padding,dilation,groups):
    """N-dimensional cross-correlation, including grouped 1D and vision 3D."""
    d=len(x.shape)-2
    if d<1 or len(w.shape)!=d+2 or groups<1:raise ValueError('CONV_RANK_GROUPS')
    if any(len(v)!=d for v in (stride,padding,dilation)):raise ValueError('CONV_OPTIONS')
    if any(s<=0 for s in stride+dilation) or any(p<0 for p in padding):raise ValueError('CONV_OPTIONS')
    B,C,*spatial=x.shape;O,I,*kernel=w.shape
    if C%groups or O%groups or I!=C//groups:raise ValueError('CONV_CHANNELS')
    if bias is not None and bias.shape!=(O,):raise ValueError('CONV_BIAS')
    size=tuple((n+2*p-di*(k-1)-1)//s+1 for n,p,di,k,s in zip(spatial,padding,dilation,kernel,stride))
    if any(n<=0 for n in size):raise ValueError('CONV_OUTPUT_SIZE')
    out=[]
    for n in range(B):
        for o in range(O):
            g=o//(O//groups)
            for loc in indices(size):
                value=0.0
                for c in range(I):
                    for ki in indices(kernel):
                        xi=tuple(v*s-p+i*di for v,s,p,i,di in zip(loc,stride,padding,ki,dilation))
                        if all(0<=i<sz for i,sz in zip(xi,spatial)):
                            value += x.at((n,g*I+c)+xi)*w.at((o,c)+ki)
                if bias is not None:value+=bias.data[o]
                out.append(value)
    return Array((B,O)+size,tuple(out))

def layernorm(x,normalized_shape,weight,bias,eps):
    k=len(normalized_shape)
    if k==0 or tuple(x.shape[-k:])!=tuple(normalized_shape):raise ValueError('LN_SHAPE')
    D=math.prod(normalized_shape)
    if D==0:raise ValueError('LN_EMPTY_AXIS')
    if weight is not None and weight.shape!=tuple(normalized_shape):raise ValueError('LN_WEIGHT')
    if bias is not None and bias.shape!=tuple(normalized_shape):raise ValueError('LN_BIAS')
    out=[];means=[];rs=[]
    for start in range(0,len(x.data),D):
        row=x.data[start:start+D];avg=sum(row)/D
        variance=sum((v-avg)**2 for v in row)/D;r=rsqrt(variance+eps)
        means.append(avg);rs.append(r)
        for i,v in enumerate(row):out.append((v-avg)*r*(weight.data[i] if weight is not None else 1)+(bias.data[i] if bias is not None else 0))
    rshape=x.shape[:-k]+(1,)*k
    return Array(x.shape,tuple(out)),Array(rshape,tuple(means)),Array(rshape,tuple(rs))

def cumsum(x,dim):
    dim=dim_axis(dim,len(x.shape));out=[]
    for ix in indices(x.shape):
        total=0
        for j in range(ix[dim]+1):total+=x.at(ix[:dim]+(j,)+ix[dim+1:])
        out.append(total)
    return Array(x.shape,tuple(out))

def sigmoid(x):
    if x>=0:return 1/(1+exp(-x))
    e=exp(x);return e/(1+e)
def exp(x):
    try:return math.exp(x)
    except OverflowError:return math.inf

def sqrt(x):return math.sqrt(x) if x>=0 else math.nan

def rsqrt(x):
    if x==0:return math.copysign(math.inf,x)
    return 1/math.sqrt(x) if x>0 else math.nan

def log(x):
    if x==0:return -math.inf
    return math.log(x) if x>0 else math.nan

def divide(a,b):
    if b==0:
        if a==0 or math.isnan(a):return math.nan
        return math.copysign(math.inf,math.copysign(1,a)*math.copysign(1,b))
    return a/b

def silu(x):return x*sigmoid(x)
def gelu(x,approximate='none'):
    if approximate=='none':return x*.5*(1+math.erf(x/math.sqrt(2)))
    if approximate=='tanh':return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))
    raise ValueError('GELU_APPROXIMATE')

def softplus(x,beta=1.,threshold=20.):
    if beta<=0:raise ValueError('SOFTPLUS_BETA')
    z=beta*x
    return x if z>threshold else math.log1p(exp(z))/beta


def embedding_backward(grad,ids,num_weights,padding_idx,scale_grad_by_freq):
    if grad.shape[:-1]!=ids.shape:raise ValueError('EMBEDDING_GRAD_SHAPE')
    width=grad.shape[-1];out=[0.0]*(num_weights*width);counts={}
    for ident in ids.data:counts[ident]=counts.get(ident,0)+1
    for i,ident in enumerate(ids.data):
        if not 0<=ident<num_weights:raise IndexError('EMBEDDING_ID')
        if ident==padding_idx:continue
        scale=counts[ident] if scale_grad_by_freq else 1
        for d in range(width):out[ident*width+d]+=grad.data[i*width+d]/scale
    return Array((num_weights,width),tuple(out))


def layernorm_backward(grad,x,norm_shape,mean,rstd,weight,bias,mask):
    n=math.prod(norm_shape);m=math.prod(x.shape)//n
    gx=[];gw=[0.0]*n;gb=[0.0]*n
    for row in range(m):
        xs=x.data[row*n:(row+1)*n];gs=grad.data[row*n:(row+1)*n]
        z=[(xx-mean.data[row])*rstd.data[row] for xx in xs]
        g=[dy*(weight.data[i] if weight is not None else 1) for i,dy in enumerate(gs)]
        s=sum(g);sz=sum(a*b for a,b in zip(g,z))
        gx.extend(rstd.data[row]*(g[i]-s/n-z[i]*sz/n) for i in range(n))
        for i in range(n):gw[i]+=gs[i]*z[i];gb[i]+=gs[i]
    return (Array(x.shape,tuple(gx)) if mask[0] else None,
            Array(norm_shape,tuple(gw)) if mask[1] else None,
            Array(norm_shape,tuple(gb)) if mask[2] else None)


def convolution_backward(grad,x,w,stride,padding,dilation,groups,mask):
    ndim=len(x.shape)-2;batch,cin=x.shape[:2];cout=w.shape[0];kin=w.shape[1];kshape=w.shape[2:];osp=grad.shape[2:]
    if cin!=groups*kin or cout%groups:raise ValueError('CONV_BACKWARD_GROUPS')
    gx=[0.0]*len(x.data);gw=[0.0]*len(w.data);gb=[0.0]*cout
    def offset(ix,shape):
        pos=0
        for i,n in zip(ix,shape):pos=pos*n+i
        return pos
    for n in range(batch):
        for oc in range(cout):
            group=oc//(cout//groups)
            for oi in indices(osp):
                dy=grad.at((n,oc)+oi);gb[oc]+=dy
                for kic in range(kin):
                    ic=group*kin+kic
                    for ki in indices(kshape):
                        xi=tuple(oi[d]*stride[d]-padding[d]+ki[d]*dilation[d] for d in range(ndim))
                        if not all(0<=i<dim for i,dim in zip(xi,x.shape[2:])):continue
                        wi=(oc,kic)+ki;ix=(n,ic)+xi
                        gx[offset(ix,x.shape)]+=dy*w.at(wi)
                        gw[offset(wi,w.shape)]+=dy*x.at(ix)
    return (Array(x.shape,tuple(gx)) if mask[0] else None,
            Array(w.shape,tuple(gw)) if mask[1] else None,
            Array((cout,),tuple(gb)) if mask[2] else None)


def log1p(x):
    if x == -1:return -math.inf
    return math.log1p(x) if x > -1 else math.nan


def finite_trig(x, function):
    return function(x) if math.isfinite(x) else math.nan


def integral_round(x, function):
    if not math.isfinite(x):return x
    y=function(x)
    return math.copysign(0.0,x) if y==0 else y


def power(a,b):
    """Real floating power with explicit domain/overflow results, not Python complex.

    Python/libm is the declared primitive. This normalizes ValueError/overflow to
    real IEEE-style NaN/Infinity used by the tensor power path. Integer tensor
    modular arithmetic is handled separately at the typed dispatch boundary.
    """
    try:
        return math.pow(a,b)
    except OverflowError:
        odd=math.isfinite(b) and b==math.floor(b) and b%2!=0
        return -math.inf if a<0 and odd else math.inf
    except ValueError:
        if a==0 and b<0:
            odd=math.isfinite(b) and b==math.floor(b) and b%2!=0
            return -math.inf if math.copysign(1.,a)<0 and odd else math.inf
        return math.nan


def integer_divide(a, b, mode):
    """Integer quotient without conversion through binary64."""
    if b==0:raise ZeroDivisionError('INTEGER_DIVISION_BY_ZERO')
    if mode=='floor':return a//b
    if mode=='trunc':return (-1 if (a<0)!=(b<0) else 1)*(abs(a)//abs(b))
    raise ValueError('INTEGER_DIVISION_MODE')


def rounded_divide(a, b, mode):
    """Floating quotient/trunc or CPython-style floating floor divmod rule.

    Keep nonfinite results in the arithmetic domain, not math.floor(Inf).
    Infinity divisor and signed zero are intentional tested boundary cases.
    """
    if mode=='trunc':
        quotient=divide(a,b)
        return integral_round(quotient,math.trunc)
    if mode!='floor':raise ValueError('FLOAT_DIVISION_MODE')
    if b==0:return divide(a,b)
    try:mod=math.fmod(a,b)
    except ValueError:return math.nan
    result=divide(a-mod,b)
    if mod!=0 and (b<0)!=(mod<0):result-=1.0
    if result!=0:
        if not math.isfinite(result):return result
        floored=math.floor(result)
        return floored+1 if result-floored>0.5 else floored
    return math.copysign(0.0,divide(a,b))
