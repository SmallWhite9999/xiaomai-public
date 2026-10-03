"""Python integer-coordinate arange. No arange kernel; exact integer arithmetic.

Fractional and nonfinite start/end/step are not admitted. Integer-valued floats
are accepted; floating output coordinates are bounded to the exact float32 range.
"""
from __future__ import annotations
import torch
import math
from . import kernels as k
from .numeric_words import encode_scalar
OPERATORS=frozenset('aten.'+n for n in ('arange.default','arange.start','arange.start_step','arange.start_out'))

def execute(name,v,*,sink=None,progress=None):
    if name not in OPERATORS:raise k.StorageError('SEQUENCE_OPERATOR_NOT_IMPLEMENTED')
    start,end,step=v.get('start',0),v['end'],v.get('step',1)
    had_float=any(type(x) is float for x in (start,end,step))
    if any(type(x) not in (int,float) or type(x) is float and (not math.isfinite(x) or not x.is_integer()) for x in (start,end,step)):
        raise k.StorageError('ARANGE_INTEGER_VALUED_ENDPOINTS_REQUIRED')
    start,end,step=map(int,(start,end,step))
    if step==0 or start<end and step<0 or start>end and step>0:raise k.StorageError('ARANGE_INVALID_STEP')
    count=len(range(start,end,step));out=v.get('out')
    dtype=(out.dtype if out is not None else v.get('dtype')) or (torch.get_default_dtype() if had_float else torch.int64)
    if dtype is torch.bool:raise k.StorageError('ARANGE_BOOL_NOT_ADMITTED')
    if dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64) and any(abs(x)>2**24 for x in (start,end,step)):
        raise k.StorageError('ARANGE_LARGE_FLOAT_COORDINATES_NOT_ADMITTED')
    if out is not None:
        k.admit(out)
        if tuple(out.shape)!=(count,) or not k.nonoverlap(out):raise k.StorageError('ARANGE_OUT_RESIZE_OR_OVERLAP_NOT_ADMITTED')
    else:
        out=k.allocate((count,),dtype=dtype,**{key:v.get(key) for key in ('device','layout','pin_memory')})
    target=k.raw(out);width=out.element_size()
    for j,value in enumerate(range(start,end,step)):
        if j%4096==0:k.tick(progress,operation=name,element=j)
        bits=encode_scalar(value,dtype);at=k.offset(out,(j,))*width
        k.emit(sink,'STORAGE_WRITE_INTENT',operation=name,dst_byte=at,
            before_hex=bytes(target[at:at+width]).hex() if v.get('out') is not None else None,
            scalar_integer=value,after_hex=bits.hex(),fresh_allocation=v.get('out') is None)
        target[at:at+width]=bits
        k.emit(sink,'STORAGE_WRITE_DONE',operation=name,dst_byte=at,byte_count=width)
    return out
