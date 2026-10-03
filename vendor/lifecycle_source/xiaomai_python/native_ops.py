"""Python index algorithms for CPU padding and slice/select gradient placement.

Coordinates, crop rules, and source/destination offsets are computed in Python.
Native dependencies are tensor allocation, scalar-to-dtype conversion and raw
storage access. No pad/select_backward/slice_backward kernel performs the work.
Copies preserve original scalar bits (including NaN payloads and large integers).
Temporary byte buffers are bounded; source tensors are never converted to tuples.
"""
from __future__ import annotations
import ctypes
import itertools
import torch

OPERATORS=frozenset(('aten.constant_pad_nd.default','aten.select_backward.default',
                     'aten.slice_backward.default'))
BLOCK_BYTES=65536
DTYPES=(torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
        torch.float16,torch.bfloat16,torch.float32,torch.float64)


def _admit(tensor):
    if not any(type(tensor) is t for t in (torch.Tensor,torch.nn.Parameter)):
        raise ValueError('PYTHON_INDEX_TENSOR_SUBCLASS_UNSUPPORTED')
    if (tensor.device.type!='cpu' or tensor.layout!=torch.strided or tensor.is_quantized
            or tensor.is_complex() or tensor.is_conj() or tensor.is_neg()):
        raise ValueError('PYTHON_INDEX_CPU_PLAIN_REAL_STRIDED_ONLY')
    if not any(tensor.dtype is dtype for dtype in DTYPES):raise ValueError('PYTHON_INDEX_DTYPE_NOT_ADMITTED')
    if any(s<0 for s in tensor.stride()):raise ValueError('PYTHON_INDEX_NEGATIVE_STRIDE_UNSUPPORTED')


def _integers(values, message):
    if not any(type(values) is t for t in (list,tuple)) or any(type(x) is not int for x in values):
        raise ValueError(message)
    return tuple(values)


def _dim(dim, rank):
    if type(dim) is not int or not -rank<=dim<rank:raise ValueError('PYTHON_INDEX_DIM_OUT_OF_RANGE')
    return dim%rank


def _storage(tensor):
    storage=tensor.untyped_storage()
    if not storage.nbytes():return memoryview(bytearray())
    return memoryview((ctypes.c_ubyte*storage.nbytes()).from_address(storage.data_ptr())).cast('B')


def _offset(tensor, coords):
    return tensor.storage_offset()+sum(i*s for i,s in zip(coords,tensor.stride()))


def _suggest_memory_format(source):
    """Python metadata equivalent of eager suggest_memory_format.

    Unlike channels-last contiguity, this also checks singleton-axis strides
    and batch/channel ambiguity. Mirrors c10/core/MemoryFormat.h as exposed by
    torch._prims_common.are_strides_like_channels_last_or_false; no tensor
    arithmetic, copying or padding kernel is used to choose the format.
    """
    if source.ndim==4:order=(1,3,2,0);result=torch.channels_last
    elif source.ndim==5:order=(1,4,3,2,0);result=torch.channels_last_3d
    else:return torch.contiguous_format
    shape=source.shape;strides=source.stride()
    if strides[1]==0:return torch.contiguous_format
    minimum=0
    for dim in order:
        if shape[dim]==0 or strides[dim]<minimum:return torch.contiguous_format
        if dim==0 and minimum==strides[1]:return torch.contiguous_format
        minimum=strides[dim]*shape[dim]
    return result


def _new(shape, source, *, preserve_format=False):
    memory_format=_suggest_memory_format(source) if preserve_format else torch.contiguous_format
    return torch.empty(shape,dtype=source.dtype,device='cpu',memory_format=memory_format)


def _fill(tensor, value, *, operator, record=None, progress=None):
    if not any(type(value) is t for t in (bool,int,float)):raise ValueError('PYTHON_INDEX_REAL_FILL_SCALAR_REQUIRED')
    # This is a declared dtype/storage conversion, not a native padding kernel.
    scalar=torch.tensor(value,dtype=tensor.dtype,device='cpu')
    pattern=bytes(_storage(scalar))
    target=_storage(tensor);size=tensor.element_size()
    block_elements=max(1,BLOCK_BYTES//size)
    for begin in range(0,tensor.numel(),block_elements):
        count=min(block_elements,tensor.numel()-begin)
        if progress:progress.work(operator=operator,index_stage='fill',element_begin=begin,element_count=count)
        target[begin*size:(begin+count)*size]=pattern*count
        if record:
            for i in range(begin,begin+count):
                record('SCALAR_INDEX_FILL',operator=operator,target_storage_element=i,value_bits=pattern.hex())


def _copy(source, target, shape, source_index, target_index, *, operator, record=None, progress=None):
    if any(n==0 for n in shape):return
    source_bytes,target_bytes=_storage(source),_storage(target)
    size=source.element_size()
    width=shape[-1] if shape else 1
    outer=itertools.product(*(range(n) for n in shape[:-1])) if shape else [()]
    for prefix in outer:
        coord=prefix+(0,) if shape else ()
        src=_offset(source,source_index(coord));dst=_offset(target,target_index(coord))
        if width>1:
            next_coord=prefix+(1,)
            src_step=_offset(source,source_index(next_coord))-src
            dst_step=_offset(target,target_index(next_coord))-dst
        else:src_step=dst_step=1
        step=max(1,BLOCK_BYTES//size)
        for start in range(0,width,step):
            count=min(step,width-start)
            if progress:progress.work(operator=operator,index_stage='copy',row=list(prefix),element_begin=start,element_count=count)
            if src_step==dst_step==1 and not record:
                a=(src+start)*size;b=(dst+start)*size
                target_bytes[b:b+count*size]=source_bytes[a:a+count*size]
            else:
                for j in range(start,start+count):
                    a=(src+j*src_step)*size;b=(dst+j*dst_step)*size
                    bits=source_bytes[a:a+size]
                    target_bytes[b:b+size]=bits
                    if record:record('SCALAR_INDEX_COPY',operator=operator,
                        source_storage_element=src+j*src_step,target_storage_element=dst+j*dst_step,value_bits=bytes(bits).hex())


def constant_pad_nd(source, pad, value=0, *, record=None, progress=None):
    _admit(source);pad=_integers(pad,'PYTHON_PAD_INTEGER_PAIRS_REQUIRED')
    if len(pad)%2 or len(pad)>2*source.ndim:raise ValueError('PYTHON_PAD_RANK_OR_PAIRS')
    shape=list(source.shape);crop=[0]*source.ndim;dest=[0]*source.ndim;region=list(shape)
    for pair in range(len(pad)//2):
        dim=source.ndim-1-pair;left,right=pad[2*pair:2*pair+2]
        crop[dim]=max(0,-left);dest[dim]=max(0,left)
        region[dim]+=min(left,0)+min(right,0)
        shape[dim]+=left+right
        if region[dim]<0 or shape[dim]<0:raise ValueError('PYTHON_PAD_CROP_EXCEEDS_SOURCE')
    operator='aten.constant_pad_nd.default'
    with torch.no_grad(),torch._C._DisableTorchDispatch():
        if any(p>0 for p in pad):
            result=_new(shape,source,preserve_format=True)
            _fill(result,value,operator=operator,record=record,progress=progress)
        else:
            # Native allocation/layout rules retain the cropped clone's stride
            # order; all actual element placement below is still Python-owned.
            template=torch.as_strided(source,region,source.stride(),_offset(source,crop))
            result=torch.empty_like(template,memory_format=torch.preserve_format)
        _copy(source,result,region,lambda ix:tuple(i+c for i,c in zip(ix,crop)),
            lambda ix:tuple(i+d for i,d in zip(ix,dest)),operator=operator,record=record,progress=progress)
    return result


def _backward_shape(grad, input_sizes, dim):
    _admit(grad);shape=_integers(input_sizes,'PYTHON_INDEX_INTEGER_INPUT_SIZES_REQUIRED')
    if any(n<0 for n in shape):raise ValueError('PYTHON_INDEX_NEGATIVE_INPUT_SIZE')
    return shape,_dim(dim,len(shape))


def select_backward(grad, input_sizes, dim, index, *, record=None, progress=None):
    shape,dim=_backward_shape(grad,input_sizes,dim)
    if type(index) is not int or not -shape[dim]<=index<shape[dim]:raise ValueError('PYTHON_SELECT_INDEX_OUT_OF_RANGE')
    index%=shape[dim]
    if tuple(grad.shape)!=shape[:dim]+shape[dim+1:]:raise ValueError('PYTHON_SELECT_GRAD_SHAPE_MISMATCH')
    operator='aten.select_backward.default'
    with torch.no_grad(),torch._C._DisableTorchDispatch():
        result=_new(shape,grad)
        _fill(result,0,operator=operator,record=record,progress=progress)
        _copy(grad,result,tuple(grad.shape),lambda ix:ix,lambda ix:ix[:dim]+(index,)+ix[dim:],
            operator=operator,record=record,progress=progress)
    return result


def slice_backward(grad, input_sizes, dim, start, end, step, *, record=None, progress=None):
    shape,dim=_backward_shape(grad,input_sizes,dim)
    if any(type(x) is not int for x in (start,end,step)) or step<=0:raise ValueError('PYTHON_SLICE_POSITIVE_INTEGER_STEP_REQUIRED')
    start,end,step=slice(start,end,step).indices(shape[dim])
    expected=list(shape);expected[dim]=len(range(start,end,step))
    if tuple(grad.shape)!=tuple(expected):raise ValueError('PYTHON_SLICE_GRAD_SHAPE_MISMATCH')
    operator='aten.slice_backward.default'
    with torch.no_grad(),torch._C._DisableTorchDispatch():
        result=_new(shape,grad)
        _fill(result,0,operator=operator,record=record,progress=progress)
        _copy(grad,result,tuple(grad.shape),lambda ix:ix,
            lambda ix:ix[:dim]+(start+ix[dim]*step,)+ix[dim+1:],operator=operator,record=record,progress=progress)
    return result


def execute(name, values, *, record=None, progress=None):
    if name=='aten.constant_pad_nd.default':
        return constant_pad_nd(values['self'],values['pad'],values['value'],record=record,progress=progress)
    if name=='aten.select_backward.default':
        return select_backward(values['grad_output'],values['input_sizes'],values['dim'],values['index'],record=record,progress=progress)
    if name=='aten.slice_backward.default':
        return slice_backward(values['grad_output'],values['input_sizes'],values['dim'],values['start'],values['end'],values['step'],record=record,progress=progress)
    raise ValueError('PYTHON_INDEX_OPERATOR_UNSUPPORTED:'+name)
