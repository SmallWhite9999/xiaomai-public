"""Install scalar Python arithmetic at actual ATen calls of a caller-owned model.

No second model or surrogate predictions are used. Strict mode rejects every
unlisted operator; an explicit structural/native allowlist is always reported.
Autograd scheduling, storage/layout, casts and libm are declared dependencies.
Keep this mode active for backward too to observe its dispatched arithmetic.
"""
from __future__ import annotations
from xiaomai_observe.passive import type_is_one_of
from contextlib import contextmanager
import contextvars
import hashlib
import inspect
import math
from pathlib import Path
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from xiaomai_observe.capture import Capture, BUSY, measurement, source_identity
from xiaomai_observe.journal import EvidenceError
from xiaomai_observe.runtime import _SCOPE
from xiaomai_observe.passive import actual_isinstance
from . import core, training_ops, selection_ops, native_ops, streaming_mm
from xiaomai_storage import (kernels as storage_ops, index_ops as storage_index_ops,
    view_ops, numeric_words, placement_ops, sequence_ops, bridge_ops, allocation)

class BackendError(EvidenceError): pass

# These operators retain the original storage/layout/cast/index implementation.
# They are NOT advertised as rewritten Python arithmetic.
STRUCTURAL = frozenset('''aten.view.default aten._unsafe_view.default aten.reshape.default
aten.permute.default aten.transpose.int aten.t.default aten.unsqueeze.default
aten.squeeze.default aten.squeeze.dim aten.squeeze.dims aten.expand.default
aten.as_strided.default aten.slice.Tensor aten.select.int aten.split.Tensor
aten.split_with_sizes.default aten.unbind.int aten.cat.default aten.stack.default
aten.clone.default aten.detach.default aten.alias.default aten.copy_.default
aten.contiguous.default aten._to_copy.default aten.to.dtype aten.to.device
aten.empty.memory_format aten.empty_strided.default aten.empty_like.default
aten.zeros.default aten.zeros_like.default aten.ones.default aten.ones_like.default
aten.full.default aten.full_like.default aten.fill_.Scalar aten.zero_.default
aten.lift_fresh.default aten.lift_fresh_copy.default aten.scalar_tensor.default
aten.eye.default aten.eye.m aten.arange.default aten.arange.start aten.arange.start_step aten.arange.start_out
aten.new_empty.default aten.new_empty_strided.default aten.new_zeros.default aten.new_full.default
aten.index.Tensor aten.index_select.default aten.scatter.src
aten.scatter_.src aten.index_put_.default aten.index_put.default
aten.slice_scatter.default aten.select_scatter.default
aten.repeat.default aten.repeat_interleave.self_int aten.flip.default
aten._local_scalar_dense.default aten.is_nonzero.default aten.nonzero.default
aten.is_same_size.default aten._has_same_storage_numel.default
aten._pin_memory.default aten.pin_memory.default aten.lift.default
profiler._record_function_enter_new.default profiler._record_function_exit._RecordFunction
profiler._record_function_exit.default'''.split())

_FLOATS=(torch.float16,torch.bfloat16,torch.float32,torch.float64)

def array(t):
    if actual_isinstance(t,core.Array): return t
    if actual_isinstance(t,torch.Tensor):
        if not type_is_one_of(type(t),(torch.Tensor,torch.nn.Parameter)):raise BackendError('TENSOR_SUBCLASS_UNSUPPORTED')
        if t.device.type!='cpu' or t.layout!=torch.strided or t.is_quantized or t.is_complex():
            raise BackendError('REFERENCE_BACKEND_CPU_REAL_STRIDED_ONLY')
        storage_ops.admit(t)
        source=storage_ops.raw(t);width=t.element_size()
        values=[]
        for ix in storage_ops.coords(t.shape):
            start=storage_ops.offset(t,ix)*width
            values.append(numeric_words.decode(bytes(source[start:start+width]),t.dtype))
        return core.Array(tuple(t.shape),tuple(values))
    if type_is_one_of(type(t),(int,float,bool)):return core.scalar(t)
    raise BackendError('NON_NUMERIC_ARGUMENT')

def tensor(a,*,dtype,device='cpu'):
    # Integer arithmetic uses Python's exact accumulator, then the declared
    # tensor width's modular conversion; never round through a float.
    if dtype in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
        info=torch.iinfo(dtype);modulus=1<<info.bits
        def convert(z):
            if type(z) is not int and type(z) is not bool:
                raise BackendError('INTEGER_RESULT_WOULD_REQUIRE_LOSSY_FLOAT_CAST')
            y=int(z)%modulus
            return y-modulus if y>info.max else y
        a=core.Array(a.shape,tuple(convert(z) for z in a.data))
    result=storage_ops.allocate(a.shape,dtype=dtype,device=device,memory_format=torch.contiguous_format)
    data=storage_ops.raw(result);width=result.element_size()
    for i,value in enumerate(a.data):
        data[i*width:(i+1)*width]=numeric_words.encode_scalar(value,dtype)
    return result

def schema_arguments(func,args,kwargs):
    values=dict(kwargs)
    for spec,value in zip(func._schema.arguments,args):
        if spec.name in values:raise BackendError('DUPLICATE_ARGUMENT')
        values[spec.name]=value
    for spec in func._schema.arguments:
        if spec.name not in values and spec.has_default_value():values[spec.name]=spec.default_value
    return values

def dtype_for(x,y=None):
    if y is not None:
        with torch._C._DisableTorchDispatch():return torch.result_type(x,y)
    return x.dtype if actual_isinstance(x,torch.Tensor) else torch.get_default_dtype()

def _allclose(v):
    """CPU FP32/FP64 nondifferentiable predicate; no native comparison kernel."""
    left,right=v['self'],v['other']
    if (not type_is_one_of(type(left),(torch.Tensor,torch.nn.Parameter)) or
            not type_is_one_of(type(right),(torch.Tensor,torch.nn.Parameter))):
        raise BackendError('ALLCLOSE_ORDINARY_TENSORS_REQUIRED')
    if left.dtype is not right.dtype or left.dtype not in (torch.float32,torch.float64):
        raise BackendError('ALLCLOSE_MATCHING_FP32_FP64_REQUIRED')
    rtol,atol=v['rtol'],v['atol']
    if any(type(value) not in (int,float) or not value>=0 for value in (rtol,atol)):
        raise BackendError('ALLCLOSE_NONNEGATIVE_TOLERANCES_REQUIRED')
    if type(v['equal_nan']) is not bool:raise BackendError('ALLCLOSE_BOOL_EQUAL_NAN_REQUIRED')
    a,b=array(left),array(right);shape=core.broadcast_shape(a.shape,b.shape)
    dtype=left.dtype
    def R(value):return numeric_words.decode(numeric_words.encode_scalar(value,dtype),dtype)
    relative,absolute=R(rtol),R(atol)
    for coordinate in core.indices(shape):
        x,y=core.bat(a,coordinate),core.bat(b,coordinate)
        if x==y or (v['equal_nan'] and math.isnan(x) and math.isnan(y)):continue
        error=abs(R(x-y));allowed=R(absolute+R(relative*abs(y)))
        if not math.isfinite(error) or not error<=allowed:return False
    return True


# Each kernel returns an Array or a tuple of Arrays; public return tensors are
# created by _compute. This makes the arithmetic independently executable.
def _compute(name,v,emit):
    if name=='aten.allclose.default':return _allclose(v),None
    if name in EXTRA_OPERATORS:
        return _extra(name,v,emit)
    x=v.get('self',v.get('input'));dt=dtype_for(x) if x is not None else None
    if name=='aten.mm.default':return core.mm(array(v['self']),array(v['mat2']),record=emit),dt
    if name=='aten.bmm.default':return core.bmm(array(x),array(v['mat2']),record=emit),dt
    if name in ('aten.addmm.default','aten.baddbmm.default'):
        # beta==0 deliberately ignores NaN/Inf in the addend, matching contract.
        fn=core.mm if name=='aten.addmm.default' else core.bmm
        p=fn(array(v['mat1'] if 'mat1' in v else v['batch1']),array(v['mat2'] if 'mat2' in v else v['batch2']),record=emit)
        alpha=v['alpha'];beta=v['beta']
        r=core.elementwise(lambda z:alpha*z,p)
        if beta!=0:r=core.elementwise(lambda a,b:a+beta*b,r,array(x))
        return r,dt
    if name=='aten.dot.default':
        a,b=array(x),array(v['tensor']);
        if len(a.shape)!=1 or a.shape!=b.shape:raise BackendError('DOT_SHAPE')
        return core.scalar(sum(i*j for i,j in zip(a.data,b.data))),dt
    if name=='aten.bitwise_not.default':
        if dt not in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
            raise BackendError('BITWISE_REQUIRES_BOOL_OR_INTEGER')
        op=(lambda z:not z) if dt==torch.bool else (lambda z:(~z)&255) if dt==torch.uint8 else (lambda z:~z)
        return core.elementwise(op,array(x)),dt
    unary={
        'neg':lambda z:-z, 'abs':abs, 'sign':lambda z:int(z>0)-int(z<0), 'sgn':lambda z:int(z>0)-int(z<0), 'exp':core.exp, 'log':core.log,
        'log1p':core.log1p,'sqrt':core.sqrt,'rsqrt':core.rsqrt,
        'sin':lambda z:core.finite_trig(z,math.sin),'cos':lambda z:core.finite_trig(z,math.cos),'tanh':math.tanh,'sigmoid':core.sigmoid,
        'silu':core.silu,'square':lambda z:z*z,'reciprocal':lambda z:core.divide(1,z),
        'isnan':math.isnan,'isinf':math.isinf,'isposinf':lambda z:z==math.inf,
        'isneginf':lambda z:z==-math.inf,'isfinite':math.isfinite,
        'floor':lambda z:core.integral_round(z,math.floor),'ceil':lambda z:core.integral_round(z,math.ceil),'logical_not':lambda z:not z,
    }
    base=name.split('.')[1].removesuffix('_')
    if base in unary:
        if base in ('floor','ceil') and dt not in _FLOATS:return array(x),dt
        if base in ('exp','log','log1p','sqrt','rsqrt','sin','cos','tanh','sigmoid','reciprocal') and dt not in _FLOATS:dt=torch.get_default_dtype()
        result=core.elementwise(unary[base],array(x))
        return result,torch.bool if base.startswith('is') or base=='logical_not' else dt
    if base in ('add','sub','rsub','mul','div','pow','eq','ne','lt','le','gt','ge','maximum','minimum','logical_and','logical_or'):
        y=v.get('other',v.get('exponent'));alpha=v.get('alpha',1)
        if base=='rsub' and (dt not in _FLOATS or type(y) not in (int,float) or type(alpha) not in (int,float)):
            raise BackendError('RSUB_SCALAR_FLOATING_REAL_DOMAIN_REQUIRED')
        functions={'add':lambda a,b:a+alpha*b,'sub':lambda a,b:a-alpha*b,'rsub':lambda a,b:b-alpha*a,'mul':lambda a,b:a*b,
                   'div':core.divide,'pow':core.power,'eq':lambda a,b:a==b,'ne':lambda a,b:a!=b,
                   'lt':lambda a,b:a<b,'le':lambda a,b:a<=b,'gt':lambda a,b:a>b,'ge':lambda a,b:a>=b,
                   'maximum':lambda a,b:math.nan if math.isnan(a) or math.isnan(b) else max(a,b),
                   'minimum':lambda a,b:math.nan if math.isnan(a) or math.isnan(b) else min(a,b),
                   'logical_and':lambda a,b:bool(a) and bool(b),'logical_or':lambda a,b:bool(a) or bool(b)}
        d=dtype_for(x,y)
        if base=='div' and v.get('rounding_mode') in ('floor','trunc'):
            mode=v['rounding_mode']
            if d not in _FLOATS:
                if d==torch.bool:raise BackendError('BOOL_INTEGER_DIVISION_NOT_SUPPORTED')
                info=torch.iinfo(d)
                def op(a,b):
                    if a==info.min and b==-1:raise BackendError('INTEGER_DIVISION_OVERFLOW_NOT_ADMITTED')
                    return core.integer_divide(a,b,mode)
            else:op=lambda a,b:core.rounded_divide(a,b,mode)
            return core.elementwise(op,array(x),array(y)),d
        if base=='pow' and d not in _FLOATS:
            if d==torch.bool:raise BackendError('BOOL_POWER_NOT_SUPPORTED')
            if any(z<0 for z in array(y).data):raise BackendError('INTEGER_POWER_NEGATIVE_EXPONENT')
            info=torch.iinfo(d);modulus=2**info.bits
            def integer_power(a,b):
                result=pow(a,b,modulus)
                return result-modulus if result>info.max else result
            r=core.elementwise(integer_power,array(x),array(y))
        elif base=='pow' and name=='aten.pow.Tensor_Scalar' and y in (0.5,-0.5,-1.,2.,3.):
            # ATen's CPU scalar-exponent specializations have distinct IEEE
            # behavior (e.g. sqrt(-Inf), rsqrt(-0)); do not merge with pow(T,T).
            specialized={0.5:core.sqrt,-0.5:core.rsqrt,-1.:lambda z:core.divide(1,z),
                         2.:lambda z:z*z,3.:lambda z:z*z*z}
            r=core.elementwise(specialized[y],array(x))
        else:r=core.elementwise(functions[base],array(x),array(y))
        if base in ('eq','ne','lt','le','gt','ge','logical_and','logical_or'):d=torch.bool
        if base=='div':
            rounding=v.get('rounding_mode')
            if rounding is not None:raise BackendError('DIV_ROUNDING_MODE')
            if d not in _FLOATS:d=torch.get_default_dtype()
        return r,d
    if base in ('sum','mean','amax','amin'):
        dims=v.get('dim');dims=[dims] if actual_isinstance(dims,int) else dims
        r=core.reduce(array(x),dims,mean=base=='mean',keepdim=v.get('keepdim',False),maximum=base in ('amax','amin')) if base!='amin' else core.elementwise(lambda z:-z,core.reduce(core.elementwise(lambda z:-z,array(x)),dims,keepdim=v.get('keepdim',False),maximum=True))
        d=v.get('dtype') or dt
        if base=='sum' and v.get('dtype') is None and d not in _FLOATS:d=torch.int64
        return r,d
    if base in ('all','any'):
        a=array(x);dims=v.get('dim')
        if dims is None:r=core.scalar(all(a.data) if base=='all' else any(a.data))
        else:
            aa=core.elementwise(lambda z:int(bool(z)),a)
            dims=[dims] if actual_isinstance(dims,int) else dims
            sums=core.reduce(aa,dims,keepdim=v.get('keepdim',False))
            n=math.prod(a.shape[core.dim_axis(d,len(a.shape))] for d in dims)
            r=core.elementwise(lambda z:z==n if base=='all' else z>0,sums)
        return r,torch.bool
    if base=='cumsum':return core.cumsum(array(x),v['dim']),v.get('dtype') or (dt if dt in _FLOATS else torch.int64)
    if base in ('_softmax','_log_softmax','_safe_softmax'):
        r=core.softmax(array(x),v['dim'],log=base=='_log_softmax',safe_all_masked=base=='_safe_softmax')
        d=v.get('dtype') or (torch.float32 if v.get('half_to_float',False) else dt)
        return r,d
    if base=='_softmax_backward_data':
        g,p=array(v['grad_output']),array(v['output']);dim=v['dim']
        s=core.reduce(core.elementwise(lambda a,b:a*b,g,p),[dim],keepdim=True)
        return core.elementwise(lambda a,b,c:b*(a-c),g,p,s),v['input_dtype']
    if base=='_log_softmax_backward_data':
        g,p=array(v['grad_output']),array(v['output']);s=core.reduce(g,[v['dim']],keepdim=True)
        return core.elementwise(lambda a,b,c:a-core.exp(b)*c,g,p,s),v['input_dtype']
    if base=='embedding_dense_backward':
        return core.embedding_backward(array(v['grad_output']),array(v['indices']),v['num_weights'],v['padding_idx'],v['scale_grad_by_freq']),v['grad_output'].dtype
    if base=='native_layer_norm_backward':
        return core.layernorm_backward(array(v['grad_out']),array(v['input']),tuple(v['normalized_shape']),array(v['mean']),array(v['rstd']),None if v['weight'] is None else array(v['weight']),None if v['bias'] is None else array(v['bias']),v['output_mask']),(v['input'].dtype,v['weight'].dtype if v['weight'] is not None else v['input'].dtype,v['bias'].dtype if v['bias'] is not None else v['input'].dtype)
    if base=='convolution_backward':
        if v['transposed'] or any(v['output_padding']):raise BackendError('TRANSPOSED_CONV_BACKWARD_NOT_IMPLEMENTED')
        return core.convolution_backward(array(v['grad_output']),array(v['input']),array(v['weight']),list(v['stride']),list(v['padding']),list(v['dilation']),v['groups'],v['output_mask']),v['input'].dtype
    if base=='gelu_backward':
        def derivative(x):
            if v['approximate']=='none':return .5*(1+math.erf(x/math.sqrt(2)))+x*core.exp(-x*x/2)/math.sqrt(2*math.pi)
            if v['approximate']!='tanh':raise BackendError('GELU_APPROXIMATION')
            c=math.sqrt(2/math.pi);z=c*(x+.044715*x*x*x);t=math.tanh(z)
            return .5*(1+t)+.5*x*(1-t*t)*c*(1+3*.044715*x*x)
        return core.elementwise(lambda g,x:g*derivative(x),array(v['grad_output']),array(v['self'])),v['self'].dtype
    if base=='softplus_backward':
        return core.elementwise(lambda g,x:g if x*v['beta']>v['threshold'] else g*core.sigmoid(x*v['beta']),array(v['grad_output']),array(v['self'])),v['self'].dtype
    if base=='embedding':
        if v.get('scale_grad_by_freq') or v.get('sparse'):raise BackendError('EMBEDDING_TRAINING_OPTIONS_NOT_IMPLEMENTED')
        return core.embedding(array(v['weight']),array(v['indices'])),v['weight'].dtype
    if base=='native_layer_norm':
        affine=[t.dtype for t in (v['weight'],v['bias']) if t is not None]
        stats_dtype=affine[0] if affine else v['input'].dtype
        if any(d!=stats_dtype for d in affine) or (stats_dtype!=v['input'].dtype and not (v['input'].dtype in (torch.float16,torch.bfloat16) and stats_dtype==torch.float32)):
            raise BackendError('LAYER_NORM_MIXED_DTYPE_NOT_ADMITTED')
        return core.layernorm(array(v['input']),tuple(v['normalized_shape']),None if v['weight'] is None else array(v['weight']),None if v['bias'] is None else array(v['bias']),v['eps']),(v['input'].dtype,stats_dtype,stats_dtype)
    if base=='convolution':
        if v['transposed'] or any(v['output_padding']):raise BackendError('TRANSPOSED_CONV_NOT_IMPLEMENTED')
        r=core.convolution(array(v['input']),array(v['weight']),None if v['bias'] is None else array(v['bias']),stride=list(v['stride']),padding=list(v['padding']),dilation=list(v['dilation']),groups=v['groups'])
        return r,v['input'].dtype
    if base in ('clamp','clamp_min','clamp_max'):
        lo=v.get('min');hi=v.get('max')
        def clamp(z):
            if math.isnan(z):return z
            return min(max(z,lo) if lo is not None else z,hi) if hi is not None else (max(z,lo) if lo is not None else z)
        return core.elementwise(clamp,array(x)),dt
    if base=='where':return core.elementwise(lambda c,a,b:a if c else b,array(v['condition']),array(v['self']),array(v['other'])),dtype_for(v['self'],v['other'])
    if base=='masked_fill':return core.elementwise(lambda a,m,b:b if m else a,array(x),array(v['mask']),array(v['value'])),dt
    if base in ('tril','triu'):
        a=array(x);diag=v.get('diagonal',0)
        if len(a.shape)<2:raise BackendError('TRI_RANK')
        out=tuple(a.at(ix) if (ix[-1]-ix[-2]<=diag if base=='tril' else ix[-1]-ix[-2]>=diag) else 0 for ix in core.indices(a.shape))
        return core.Array(a.shape,out),dt
    if base=='gelu':return core.elementwise(lambda z:core.gelu(z,v['approximate']),array(x)),dt
    if base=='softplus':return core.elementwise(lambda z:core.softplus(z,v['beta'],v['threshold']),array(x)),dt
    if base=='silu_backward':return core.elementwise(lambda g,z:g*(core.sigmoid(z)+z*core.sigmoid(z)*(1-core.sigmoid(z))),array(v['grad_output']),array(v['self'])),v['self'].dtype
    if base=='sigmoid_backward':return core.elementwise(lambda g,y:g*y*(1-y),array(v['grad_output']),array(v['output'])),v['output'].dtype
    if base=='tanh_backward':return core.elementwise(lambda g,y:g*(1-y*y),array(v['grad_output']),array(v['output'])),v['output'].dtype
    if base=='nan_to_num':
        info=torch.finfo(dt);nan=v['nan'];pos=v['posinf'];neg=v['neginf']
        return core.elementwise(lambda z:(0 if nan is None else nan) if math.isnan(z) else ((info.max if pos is None else pos) if z==math.inf else ((info.min if neg is None else neg) if z==-math.inf else z)),array(x)),dt
    raise BackendError('UNIMPLEMENTED_PYTHON_OPERATOR:'+name)

def _extra(name, v, emit):
    base=name.split('.')[1].removesuffix('_'); x=v.get('self',v.get('input'))
    a=lambda x:None if x is None else array(x)
    if base in ('nll_loss_forward','nll_loss2d_forward'):
        return training_ops.nll_forward(a(x),a(v['target']),a(v['weight']),v['reduction'],v['ignore_index']),x.dtype
    if base in ('nll_loss_backward','nll_loss2d_backward'):
        return training_ops.nll_backward(a(v['grad_output']),a(x),a(v['target']),a(v['weight']),v['reduction'],v['ignore_index'],a(v['total_weight'])),x.dtype
    if base in ('mse_loss','smooth_l1_loss','mse_loss_backward','smooth_l1_loss_backward'):
        return training_ops.regression_loss(a(x),a(v['target']),v['reduction'],
            beta=v.get('beta') if base.startswith('smooth') else None,
            grad=a(v.get('grad_output'))),x.dtype
    if base in ('addcmul','addcdiv'):
        op=(lambda u,w:u*w) if base=='addcmul' else core.divide
        return core.elementwise(lambda z,u,w:z+v['value']*op(u,w),a(x),a(v['tensor1']),a(v['tensor2'])),x.dtype
    if base=='lerp':
        return core.elementwise(lambda z,y,t:z+t*(y-z),a(x),a(v['end']),a(v['weight'])),x.dtype
    if base=='linalg_vector_norm':
        return training_ops.vector_norm(a(x),v['ord'],v['dim'],v['keepdim']),v['dtype'] or x.dtype
    if base in ('sort','topk'):
        return selection_ops.select(a(x),v['dim'],descending=v['descending'] if base=='sort' else v['largest'],k=v.get('k')),(x.dtype,torch.int64)
    if base in ('max','min'):
        return selection_ops.extremum(a(x),v.get('dim'),keepdim=v.get('keepdim',False),minimum=base=='min'),((x.dtype,torch.int64) if 'dim' in v else x.dtype)
    if base in ('argmax','argmin'):
        return selection_ops.extremum(a(x),v['dim'],keepdim=v['keepdim'],minimum=base=='argmin',indices_only=True),torch.int64
    if base=='gather':
        if v.get('sparse_grad',False):raise BackendError('GATHER_SPARSE_GRAD_NOT_SUPPORTED')
        return selection_ops.gather(a(x),v['dim'],a(v['index'])),x.dtype
    if base in ('_scaled_dot_product_flash_attention_for_cpu','_scaled_dot_product_flash_attention_for_cpu_backward'):
        if v['dropout_p']!=0:raise BackendError('RANDOM_ATTENTION_DROPOUT_NOT_REWRITTEN')
        q,k,val=(a(v[n]) for n in ('query','key','value'))
        opts=dict(causal=v['is_causal'],mask=a(v['attn_mask']),scale=v['scale'])
        if base.endswith('_backward'):
            result=training_ops.attention_backward(a(v['grad_out']),q,k,val,**opts)
            return result,(v['query'].dtype,v['key'].dtype,v['value'].dtype)
        result=training_ops.attention_forward(q,k,val,**opts)
        return result[:2],(v['query'].dtype,torch.float32 if v['query'].dtype in (torch.float16,torch.bfloat16) else v['query'].dtype)
    raise BackendError('EXTRA_KERNEL_NOT_IMPLEMENTED:'+name)

EXTRA_OPERATORS=frozenset('aten.'+x for x in (
    'nll_loss_forward.default','nll_loss_backward.default','nll_loss2d_forward.default','nll_loss2d_backward.default',
    'mse_loss.default','mse_loss_backward.default','smooth_l1_loss.default','smooth_l1_loss_backward.default',
    'addcmul.default','addcmul_.default','addcdiv.default','addcdiv_.default',
    'lerp.Scalar','lerp.Tensor','lerp_.Scalar','lerp_.Tensor','linalg_vector_norm.default',
    'sort.default','sort.stable','topk.default','max.default','max.dim','min.default','min.dim',
    'argmax.default','argmin.default','gather.default',
    '_scaled_dot_product_flash_attention_for_cpu.default',
    '_scaled_dot_product_flash_attention_for_cpu_backward.default'))

_NAMES=['embedding_dense_backward.default','native_layer_norm_backward.default','convolution_backward.default','gelu_backward.default','softplus_backward.default','mm.default','bmm.default','addmm.default','baddbmm.default','dot.default','embedding.default','native_layer_norm.default','convolution.default','_softmax.default','_log_softmax.default','_safe_softmax.default','_softmax_backward_data.default','_log_softmax_backward_data.default','gelu.default','silu_backward.default','sigmoid_backward.default','tanh_backward.default','softplus.default','nan_to_num.default','nan_to_num_.default','where.self','masked_fill.Scalar','masked_fill.Tensor','masked_fill_.Scalar','tril.default','triu.default','cumsum.default','sum.default','sum.dim_IntList','mean.default','mean.dim','amax.default','amin.default','all.default','all.dim','any.default','any.dim','clamp.default','clamp_min.default','clamp_max.default']
_NAMES += [f'{x}.default' for x in ('neg','abs','sign','sgn','exp','log','log1p','sqrt','rsqrt','sin','cos','tanh','sigmoid','silu','square','reciprocal','isnan','isinf','isfinite','isposinf','isneginf','floor','ceil','logical_not')]
_NAMES += [f'{x}.{overload}' for x in ('add','sub','mul','div','eq','ne','lt','le','gt','ge') for overload in ('Tensor','Scalar')]
_NAMES += ['rsub.Scalar','allclose.default']
_NAMES += [f'{x}_.{overload}' for x in ('add','sub','mul','div') for overload in ('Tensor','Scalar')]
_NAMES += ['maximum.out','minimum.out','bitwise_not.default']
_NAMES += ['div.Tensor_mode','div.Scalar_mode','pow.Tensor_Scalar','pow.Tensor_Tensor','pow.Scalar','maximum.default','minimum.default','logical_and.default','logical_or.default']
PYTHON_OPERATORS=frozenset('aten.'+x for x in _NAMES) | EXTRA_OPERATORS
PYTHON_INDEX_OPERATORS=native_ops.OPERATORS
PYTHON_STORAGE_OPERATORS=(storage_ops.OPERATORS | storage_index_ops.OPERATORS |
    numeric_words.OPERATORS | placement_ops.OPERATORS | sequence_ops.OPERATORS | bridge_ops.DATA_OPERATORS)
PYTHON_VIEW_PLAN_OPERATORS=view_ops.OPERATORS|bridge_ops.VIEW_OPERATORS
PYTHON_ALLOCATION_OPERATORS=allocation.OPS
STRUCTURAL=STRUCTURAL-PYTHON_STORAGE_OPERATORS-PYTHON_VIEW_PLAN_OPERATORS-PYTHON_ALLOCATION_OPERATORS

class PythonReferenceBackend(TorchDispatchMode):
    def __init__(self,*,journal=None,strict=True,allowed_native=STRUCTURAL,max_input_elements=None,scalar_events=False,allow_selection_tie_policy=False,progress=None,streaming_matmul=False):
        super().__init__();self.progress=progress;self.streaming_matmul=streaming_matmul;self.journal=journal;self.strict=strict;self.allowed_native=frozenset(allowed_native)
        self.max_input_elements=max_input_elements;self.scalar_events=scalar_events
        self.allow_selection_tie_policy=allow_selection_tie_policy
        self.calls=[];self.failed=False;self.sequence=0;self.current_call=contextvars.ContextVar("python_backend_call",default=None)
        self.capture=Capture(journal,values='full',numerics=True) if journal is not None else None
        self.implementation_sha256=hashlib.sha256(Path(core.__file__).read_bytes()+Path(__file__).read_bytes()+Path(training_ops.__file__).read_bytes()+Path(selection_ops.__file__).read_bytes()+Path(streaming_mm.__file__).read_bytes()+Path(native_ops.__file__).read_bytes()+Path(storage_ops.__file__).read_bytes()+Path(storage_index_ops.__file__).read_bytes()+Path(view_ops.__file__).read_bytes()+Path(numeric_words.__file__).read_bytes()+Path(placement_ops.__file__).read_bytes()+Path(sequence_ops.__file__).read_bytes()+Path(bridge_ops.__file__).read_bytes()+Path(allocation.__file__).read_bytes()).hexdigest()
    def emit(self,kind,**fields):
        if self.journal is not None:
            with measurement():
                fields.setdefault('call_id',self.current_call.get())
                self.journal.emit(kind,backend='PYTHON_SCALAR_REFERENCE',**fields)
    def __enter__(self):
        self.emit('BACKEND_POLICY',strict=self.strict,allowed_native=sorted(self.allowed_native),python_operators=sorted(PYTHON_OPERATORS),python_index_operators=sorted(PYTHON_INDEX_OPERATORS),python_storage_operators=sorted(PYTHON_STORAGE_OPERATORS),python_view_plan_operators=sorted(PYTHON_VIEW_PLAN_OPERATORS),python_allocation_policy_operators=sorted(PYTHON_ALLOCATION_OPERATORS),implementation_sha256=self.implementation_sha256,numeric_contract='EXACT_INTEGER_THEN_WIDTH_WRAP_OR_BINARY64_SEQUENTIAL_THEN_OUTPUT_CAST',index_contract='PYTHON_COORDINATES_WITH_BIT_EXACT_STORAGE_COPY',max_input_elements=self.max_input_elements,scalar_events=self.scalar_events,native_dependencies=['tensor_allocation_and_metadata','autograd_scheduling','CPython_libm','raw_storage_byte_access','as_strided_alias_construction','optional_v082_streaming_mm_support'])
        return super().__enter__()
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:self.emit('BACKEND_CLOSED',calls=len(self.calls),failed=self.failed or args[0] is not None,whole_true_verified=False)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if BUSY.get():return func(*args,**(kwargs or {}))
        # A consuming numerical mode does not call lower TorchDispatchModes.
        # Explicitly preserve the ATen observation boundary around its real
        # implementation, including errors; never rerun the native operator.
        from xiaomai_observe.runtime import operator_observation
        with operator_observation(func,args,kwargs) as observed:
            result=self._execute_dispatch(func,types,args,kwargs)
            for call in observed:
                call['output']=result
                call['completion']='PYTHON_REFERENCE_BACKEND_RETURN_NATIVE_BOUNDARIES_DECLARED'
            return result
    def _execute_dispatch(self,func,types,args=(),kwargs=None):
        name=str(func);ordinal=self.sequence;self.sequence+=1
        v=schema_arguments(func,args,kwargs or {})
        classification='PYTHON_ALLOCATION_NATIVE_BINDING' if name in PYTHON_ALLOCATION_OPERATORS else 'PYTHON_STORAGE' if name in PYTHON_STORAGE_OPERATORS else 'PYTHON_VIEW_PLAN_NATIVE_ALIAS' if name in PYTHON_VIEW_PLAN_OPERATORS else 'PYTHON_ARITHMETIC' if name in PYTHON_OPERATORS else 'PYTHON_INDEXING' if name in PYTHON_INDEX_OPERATORS else 'DECLARED_NATIVE' if name in self.allowed_native else 'UNDECLARED'
        source=None
        frame=inspect.currentframe().f_back
        while frame:
            p=frame.f_code.co_filename.replace('\\','/')
            if '/torch/' not in p and '/xiaomai_observe/' not in p and '/xiaomai_python/dispatch.py' not in p:
                source=dict(file=frame.f_code.co_filename,line=frame.f_lineno,qualname=frame.f_code.co_qualname);break
            frame=frame.f_back
        row=dict(operator=name,ordinal=ordinal,implementation=classification,source=source,call_id=f'backend:{id(self)}:{ordinal}',parent_scope=list(_SCOPE.get()))
        self.calls.append(row)
        if self.progress:
            self.progress.operator('OPERATOR_BEGIN',operator=name,ordinal=ordinal,implementation=classification,
              tensor_shapes={k:list(t.shape) for k,t in v.items() if type_is_one_of(type(t),(torch.Tensor,torch.nn.Parameter))})
        inputs=self.capture.tree({'args':args,'kwargs':kwargs or {}},'backend.inputs') if self.capture else None
        self.emit('BACKEND_ENTER',**row,inputs=inputs)
        call_token=self.current_call.set(row['call_id'])
        start=time.perf_counter_ns()
        try:
            if classification=='UNDECLARED' and self.strict:raise BackendError('UNDECLARED_NATIVE_FALLBACK:'+name)
            if classification in ('PYTHON_STORAGE','PYTHON_VIEW_PLAN_NATIVE_ALIAS','PYTHON_ALLOCATION_NATIVE_BINDING'):
                if self.max_input_elements is not None and sum(t.numel() for t in v.values() if actual_isinstance(t,torch.Tensor))>self.max_input_elements:
                    raise BackendError('EXPLICIT_REFERENCE_RESOURCE_BOUND_EXCEEDED')
                executor=(allocation.execute if name in allocation.OPS else bridge_ops.execute if name in bridge_ops.OPERATORS else numeric_words.execute if name in numeric_words.OPERATORS else
                          placement_ops.execute if name in placement_ops.OPERATORS else
                          sequence_ops.execute if name in sequence_ops.OPERATORS else
                          view_ops.execute if name in PYTHON_VIEW_PLAN_OPERATORS else
                          storage_index_ops.execute if name in storage_index_ops.OPERATORS else storage_ops.execute)
                tick=(lambda **fields:self.progress.work(**fields)) if self.progress else None
                with torch._C._DisableTorchDispatch(),torch.no_grad():
                    result=executor(name,v,sink=self.emit if self.scalar_events else None,progress=tick)
            elif classification=='PYTHON_INDEXING':
                if self.max_input_elements is not None and sum(t.numel() for t in v.values() if actual_isinstance(t,torch.Tensor))>self.max_input_elements:
                    raise BackendError('EXPLICIT_REFERENCE_RESOURCE_BOUND_EXCEEDED')
                result=native_ops.execute(name,v,record=self.emit if self.scalar_events else None,progress=self.progress)
            elif classification!='PYTHON_ARITHMETIC':
                result=func(*args,**(kwargs or {}))
            else:
                if self.max_input_elements is not None:
                    total=sum(t.numel() for t in v.values() if actual_isinstance(t,torch.Tensor))
                    if total>self.max_input_elements:raise BackendError('EXPLICIT_REFERENCE_RESOURCE_BOUND_EXCEEDED')
                if name in ('aten.topk.default','aten.sort.default','aten.sort.stable'):
                    stable=v.get('stable') is True
                    if not stable and not self.allow_selection_tie_policy:
                        aa=array(v['self']);dim=v['dim']%len(aa.shape) if aa.shape else 0
                        for ix in core.indices(aa.shape[:dim]+aa.shape[dim+1:] if aa.shape else ()):
                            vals=[aa.at(ix[:dim]+(j,)+ix[dim:]) for j in range(aa.shape[dim])] if aa.shape else aa.data
                            if len(set(vals))!=len(vals) or any(actual_isinstance(z,float) and math.isnan(z) for z in vals):
                                raise BackendError('SELECTION_TIES_REQUIRE_EXPLICIT_DIAGNOSTIC_POLICY')
                    self.emit('SELECTION_POLICY',stable_index_ties=True,native_tie_equivalence='NOT_GUARANTEED_UNLESS_STABLE_SORT')
                emit=self.emit if self.scalar_events else None
                if self.streaming_matmul and name in ('aten.mm.default','aten.bmm.default','aten.addmm.default','aten.baddbmm.default'):
                    from . import streaming_mm
                    if name in ('aten.mm.default','aten.bmm.default'):
                        fn=streaming_mm.mm if name=='aten.mm.default' else streaming_mm.bmm
                        r=fn(v['self'],v['mat2'],record=emit,progress=self.progress);dtype=v['self'].dtype
                    else:
                        fn=streaming_mm.mm if name=='aten.addmm.default' else streaming_mm.bmm
                        r=fn(v['mat1'] if 'mat1' in v else v['batch1'],v['mat2'] if 'mat2' in v else v['batch2'],record=emit,progress=self.progress)
                        r=core.elementwise(lambda z:v['alpha']*z,r)
                        if v['beta']!=0:r=core.elementwise(lambda x,y:x+v['beta']*y,r,array(v['self']))
                        dtype=v['self'].dtype
                else:r,dtype=_compute(name,v,emit)
                result=(r if name=='aten.allclose.default' else
                    tuple(None if a is None else tensor(a,dtype=dtype[i] if actual_isinstance(dtype,tuple) else dtype) for i,a in enumerate(r)) if actual_isinstance(r,tuple) else tensor(r,dtype=dtype))
                if func._schema.is_mutable:
                    target=v.get('out',v.get('self'))
                    if not actual_isinstance(result,torch.Tensor) or not actual_isinstance(target,torch.Tensor):raise BackendError('MUTATION_SCHEMA_UNSUPPORTED')
                    if tuple(target.shape)!=tuple(result.shape):raise BackendError('OUT_RESIZE_NOT_IMPLEMENTED')
                    with torch._C._DisableTorchDispatch(),torch.no_grad():
                        storage_ops.copy(result,target,sink=emit,operation='arithmetic_writeback:'+name,
                                         progress=(lambda **fields:self.progress.work(**fields)) if self.progress else None)
                    result=target
            self.emit('BACKEND_RETURN',**row,output=self.capture.tree(result,'backend.output') if self.capture else None,elapsed_ns=time.perf_counter_ns()-start)
            if self.progress:self.progress.operator('OPERATOR_END',operator=name,ordinal=ordinal,implementation=classification,elapsed_ns=time.perf_counter_ns()-start)
            return result
        except BaseException as exc:
            self.failed=True
            if self.progress:self.progress.operator('OPERATOR_ERROR',operator=name,ordinal=ordinal,error_type=type(exc).__name__)
            self.emit('BACKEND_ERROR',**row,error_type=type(exc).__name__,error=str(exc));raise
        finally:self.current_call.reset(call_token)
