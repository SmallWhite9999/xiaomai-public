"""Python integer/IEEE encoding and strided conversion.

No torch.to/copy_/scalar_tensor/tensor conversion kernel computes values here.
Cross-dtype NaN payloads and float-to-integer overflow are explicitly rejected;
source bytes are never silently repaired. Same-dtype raw floats keep their bits.
"""
from __future__ import annotations
import math
import struct
import sys
import torch
from . import kernels as k

FLOATS={torch.float16:(5,10),torch.bfloat16:(8,7),torch.float32:(8,23),torch.float64:(11,52)}
INTS={torch.uint8:(8,False),torch.int8:(8,True),torch.int16:(16,True),torch.int32:(32,True),torch.int64:(64,True)}

def round_ratio(n,d):
    q,r=divmod(n,d)
    return q+int(r*2>d or r*2==d and q%2==1)

def encode_ratio(n,d,sign,dtype):
    eb,fb=FLOATS[dtype];bias=(1<<(eb-1))-1;emin=1-bias;emax=bias
    signbit=sign<<(eb+fb)
    if n==0:return signbit.to_bytes((1+eb+fb)//8,'little')
    exponent=n.bit_length()-d.bit_length()
    if exponent>=0:
        if n<(d<<exponent):exponent-=1
    elif (n<<(-exponent))<d:exponent-=1
    scale=fb-max(exponent,emin)
    mant=round_ratio(n<<scale,d) if scale>=0 else round_ratio(n,d<<(-scale))
    if exponent<emin:
        word=mant if mant<(1<<fb) else 1<<fb
    else:
        if mant>=(1<<(fb+1)):mant>>=1;exponent+=1
        word=((1<<eb)-1)<<fb if exponent>emax else ((exponent+bias)<<fb)+(mant-(1<<fb))
    return (signbit|word).to_bytes((1+eb+fb)//8,'little')

def encode_scalar(value,dtype,*,strict_integer=False):
    if not any(type(value) is t for t in (bool,int,float)):raise k.StorageError('BUILTIN_NUMERIC_SCALAR_REQUIRED')
    if dtype is torch.bool:return bytes((int(bool(value)),))
    if dtype in INTS:
        width,signed=INTS[dtype]
        if type(value) is float and (not math.isfinite(value)):
            raise k.StorageError('NONFINITE_TO_INTEGER_NOT_ADMITTED')
        n=int(value);lo=-(1<<(width-1)) if signed else 0;hi=(1<<(width-1))-1 if signed else (1<<width)-1
        if strict_integer:
            if type(value) is float:
                # Current ATen checked scalar conversion uses the exactly
                # representable exclusive upper bound max+1. Testing <= max
                # incorrectly rejects 127.5 -> int8, while comparing to a
                # rounded floating int64 max can admit 2**63 by mistake.
                # The lower bound remains inclusive *before* truncation.
                if value<lo or value>=hi+1:raise k.StorageError('INTEGER_SCALAR_OUT_OF_RANGE')
            else:
                lower=-hi if dtype is torch.uint8 and type(value) is int else lo
                if not lower<=n<=hi:raise k.StorageError('INTEGER_SCALAR_OUT_OF_RANGE')
        elif type(value) is float and not lo<=n<=hi:
            raise k.StorageError('INTEGER_CAST_OUT_OF_RANGE')
        return (n%(1<<width)).to_bytes(width//8,'little')
    if dtype not in FLOATS:raise k.StorageError('NUMERIC_DTYPE_NOT_ADMITTED')
    eb,fb=FLOATS[dtype]
    if strict_integer and dtype is torch.float32 and math.isfinite(value) and abs(value)>float.fromhex('0x1.fffffep+127'):
        raise k.StorageError('FLOAT32_SCALAR_OUT_OF_RANGE')
    sign=int(math.copysign(1.,value)<0) if type(value) is float else int(value<0)
    if type(value) is float and not math.isfinite(value):
        word=(sign<<(eb+fb))|(((1<<eb)-1)<<fb)
        if math.isnan(value):word|=1<<(fb-1)
        return word.to_bytes((1+eb+fb)//8,'little')
    n,d=value.as_integer_ratio() if type(value) is float else (int(value),1)
    # CPU BF16 conversion from double goes through float32. Make this boundary
    # explicit, rather than claiming direct binary64->BF16 rounding is native.
    if dtype is torch.bfloat16 or dtype is torch.float16:
        middle=encode_ratio(abs(n),d,sign,torch.float32)
        word=int.from_bytes(middle,'little')
        if dtype is torch.float16:
            middle_value=struct.unpack('<f',middle)[0]
            if math.isinf(middle_value):return ((sign<<15)|0x7c00).to_bytes(2,'little')
            middle_n,middle_d=middle_value.as_integer_ratio()
            return encode_ratio(abs(middle_n),middle_d,sign,torch.float16)
        if word&0x7f800000==0x7f800000:return (word>>16).to_bytes(2,'little')
        return (((word+0x7fff+((word>>16)&1))>>16)&0xffff).to_bytes(2,'little')
    return encode_ratio(abs(n),d,sign,dtype)

def decode(bits,dtype):
    if dtype is torch.bool:return bool(bits[0])
    if dtype in INTS:return int.from_bytes(bits,'little',signed=INTS[dtype][1])
    if dtype is torch.bfloat16:return struct.unpack('<f',b'\0\0'+bits)[0]
    fmt={torch.float16:'<e',torch.float32:'<f',torch.float64:'<d'}.get(dtype)
    if fmt is None:raise k.StorageError('NUMERIC_DTYPE_NOT_ADMITTED')
    return struct.unpack(fmt,bits)[0]

def convert(bits,source_dtype,target_dtype):
    if source_dtype is target_dtype:
        return bytes((int(bool(bits[0])),)) if target_dtype is torch.bool else bits
    value=decode(bits,source_dtype)
    if type(value) is float and math.isnan(value) and target_dtype is not torch.bool:
        raise k.StorageError('CROSS_DTYPE_NAN_PAYLOAD_NOT_ADMITTED')
    return encode_scalar(value,target_dtype)

def broadcast_index(source_shape,target_shape):
    if len(source_shape)>len(target_shape):raise k.StorageError('COPY_BROADCAST_RANK')
    for a,b in zip(reversed(source_shape),reversed(target_shape)):
        if a!=1 and a!=b:raise k.StorageError('COPY_BROADCAST_SHAPE')
    lead=len(target_shape)-len(source_shape)
    return lambda ix:tuple(0 if n==1 else ix[lead+j] for j,n in enumerate(source_shape))

def copy(source,target,*,sink=None,progress=None,operation='copy_convert',fresh=False):
    k.admit(source);k.admit(target)
    if not k.nonoverlap(target):raise k.StorageError('OVERLAPPING_DESTINATION')
    mapper=broadcast_index(tuple(source.shape),tuple(target.shape))
    a,b=k.extent(source);c,d=k.extent(target)
    same=(a==c and source.dtype is target.dtype and tuple(source.shape)==tuple(target.shape) and tuple(source.stride())==tuple(target.stride()))
    if same:
        k.emit(sink,'STORAGE_SELF_COPY',operation=operation,elements=target.numel());return target
    if a<d and c<b:raise k.StorageError('OVERLAPPING_SOURCE_DESTINATION')
    src,dst=k.raw(source),k.raw(target);sw=source.element_size();tw=target.element_size()
    if target.numel()==0:return target
    # Admission pass prevents partial writes on a late nonfinite/out-of-range
    # conversion. This is not a claim of locking external concurrent writers.
    if source.dtype is not target.dtype:
        for ordinal,ix in enumerate(k.coords(source.shape)):
            if ordinal%4096==0:k.tick(progress,operation=operation,stage='cast_validation',element=ordinal)
            at=k.offset(source,ix)*sw;bits=bytes(src[at:at+sw])
            try:convert(bits,source.dtype,target.dtype)
            except k.StorageError:
                k.emit(sink,'CAST_REJECTED',operation=operation,source_hex=bits.hex(),source_dtype=str(source.dtype),target_dtype=str(target.dtype));raise
    for ordinal,ix in enumerate(k.coords(target.shape)):
        if ordinal%4096==0:k.tick(progress,operation=operation,stage='write',element=ordinal)
        si=k.offset(source,mapper(ix))*sw;di=k.offset(target,ix)*tw
        bits=bytes(src[si:si+sw]);out=convert(bits,source.dtype,target.dtype)
        k.emit(sink,'STORAGE_WRITE_INTENT',operation=operation,src_byte=si,dst_byte=di,
               before_hex=None if fresh else bytes(dst[di:di+tw]).hex(),source_hex=bits.hex(),after_hex=out.hex(),fresh_allocation=fresh)
        dst[di:di+tw]=out
        k.emit(sink,'STORAGE_WRITE_DONE',operation=operation,dst_byte=di,byte_count=tw)
    return target

OPERATORS=frozenset(('aten._to_copy.default','aten.scalar_tensor.default'))
def execute(name,v,*,sink=None,progress=None):
    opts={key:v.get(key) for key in ('device','layout','pin_memory')}
    if name=='aten.scalar_tensor.default':
        dtype=v.get('dtype') or torch.get_default_dtype();bits=encode_scalar(v['s'],dtype,strict_integer=True)
        result=k.allocate((),dtype=dtype,**opts)
        return k.fill(result,bits,sink=sink,progress=progress,operation=name,fresh=True)
    if name=='aten._to_copy.default':
        source=k.admit(v['self']);dtype=v.get('dtype') or source.dtype
        if v.get('non_blocking',False):raise k.StorageError('NONBLOCKING_CAST_NOT_ADMITTED')
        result=k.allocate(tuple(source.shape),dtype=dtype,like=source,memory_format=v.get('memory_format'),**opts)
        if source.dtype is dtype:return k.copy(source,result,sink=sink,progress=progress,operation=name,fresh=True)
        return copy(source,result,sink=sink,progress=progress,operation=name,fresh=True)
    raise k.StorageError('NUMERIC_OPERATOR_NOT_IMPLEMENTED:'+name)
