"""Readable scalar values AND exact bits. No Base85, pickle, eval or exec.

The editable format intentionally separates an immutable export from explicit
edits. A values/bits mismatch is rejected, never silently resolved.
"""
from __future__ import annotations
import ast
import math
import struct
from pathlib import Path
import torch
from .journal import digest, canonical, new_file

DTYPES = {
    'BF16':(torch.bfloat16,2,None), 'F16':(torch.float16,2,'e'),
    'F32':(torch.float32,4,'f'), 'F64':(torch.float64,8,'d'),
    'I8':(torch.int8,1,'b'), 'U8':(torch.uint8,1,'B'),
    'I16':(torch.int16,2,'h'), 'I32':(torch.int32,4,'i'),
    'I64':(torch.int64,8,'q'), 'BOOL':(torch.bool,1,'?'),
}
TORCH_NAMES = {v[0]:k for k,v in DTYPES.items()}

def scalar_from_bits(bits: bytes, dtype: str):
    if dtype not in DTYPES or len(bits)!=DTYPES[dtype][1]:
        raise ValueError('UNSUPPORTED_DTYPE_OR_WIDTH')
    if dtype=='BF16':
        v=struct.unpack('<f',b'\0\0'+bits)[0]
    else:
        v=struct.unpack('<'+DTYPES[dtype][2],bits)[0]
    if isinstance(v,float):
        if math.isnan(v):return 'nan'
        if math.isinf(v):return '+inf' if v>0 else '-inf'
    return v

def same_readable(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,float) and a==b==0:
        return math.copysign(1,a)==math.copysign(1,b)
    return a==b

def readable_rows(raw: bytes, dtype: str, *, start:int=0):
    width=DTYPES[dtype][1]
    if len(raw)%width:raise ValueError('MISALIGNED_RAW_BYTES')
    for i in range(0,len(raw),width):
        word=raw[i:i+width]
        yield (start+i//width,scalar_from_bits(word,dtype),word.hex())

def rows_to_bytes(rows,dtype,*,start=0):
    out=bytearray()
    for expected,row in enumerate(rows,start):
        if not isinstance(row,(list,tuple)) or len(row)!=3 or row[0]!=expected:
            raise ValueError('NONCONTIGUOUS_OR_INVALID_ROW')
        word=bytes.fromhex(row[2])
        if not same_readable(row[1],scalar_from_bits(word,dtype)):
            raise ValueError('VALUE_BITS_DISAGREEMENT: use an explicit edit')
        out.extend(word)
    return bytes(out)

def tensor_bytes(t:torch.Tensor)->bytes:
    if t.layout!=torch.strided or t.device.type=='meta' or t.is_quantized:
        raise TypeError('ONLY_MATERIALIZED_DENSE_STRIDED_TENSORS_SUPPORTED')
    if t.dtype not in TORCH_NAMES:
        raise TypeError('UNSUPPORTED_DTYPE: '+str(t.dtype))
    view=t.detach()
    # BOOL copies may canonicalize any nonzero byte to 0x01. Reinterpret its
    # same-width storage before any layout copy so evidence retains raw bits.
    if view.dtype is torch.bool:view=view.view(torch.uint8)
    flat=view.contiguous().reshape(-1)
    # A length-one view can be contiguous yet retain stride 0/2/etc. Reinterpreting
    # its dtype as bytes still requires unit last stride. For at most one element,
    # changing the stride preserves every accessed offset, with no value copy that
    # could normalize noncanonical BOOL bytes or alter floating-point payloads.
    if flat.stride(0)!=1:
        if flat.numel()>1:
            raise RuntimeError('NONCOMPACT_FLAT_TENSOR')
        flat=flat.as_strided((flat.numel(),),(1,),flat.storage_offset())
    return flat.view(torch.uint8).cpu().numpy().tobytes()

def tensor_from_bytes(raw:bytes,dtype:str,shape,*,device='cpu'):
    if len(raw)!=math.prod(shape)*DTYPES[dtype][1]:raise ValueError('SHAPE_BYTES_MISMATCH')
    if not raw:return torch.empty(tuple(shape),dtype=DTYPES[dtype][0],device=device)
    # Copy/transfer bytes before applying the logical dtype. In particular, a
    # BOOL clone must not replace stored 0xff/0x80 with canonical true (0x01).
    raw_tensor=torch.frombuffer(bytearray(raw),dtype=torch.uint8).clone().to(device)
    return raw_tensor.view(DTYPES[dtype][0]).reshape(tuple(shape))

def encode_chunk(*,name:str,dtype:str,shape,start:int,raw:bytes):
    """Produce exactly the existing literal format; no file IO or evaluation."""
    header={'format':'XMO-READABLE-1','name':name,'dtype':dtype,'shape':tuple(shape),
            'start':start,'count':len(raw)//DTYPES[dtype][1], 'raw_sha256':digest(raw)}
    rows=tuple(readable_rows(raw,dtype,start=start))
    content=('# Values are readable; the third column is exact little-endian bits.\n'
             '# Do not import this data file; use the literal-only reader.\n'
             f'HEADER = {header!r}\nROWS = (\n'+''.join(f'    {r!r},\n' for r in rows)+')\n')
    return header, content.encode('utf-8')

def write_chunk(path:Path,*,name:str,dtype:str,shape,start:int,raw:bytes):
    header,content=encode_chunk(name=name,dtype=dtype,shape=shape,start=start,raw=raw)
    new_file(path,content)
    return header

def read_chunk(path:Path):
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    found={}
    for node in tree.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):
            raise ValueError('ONLY_LITERAL_ASSIGNMENTS_ALLOWED')
        key=node.targets[0].id
        if key not in ('HEADER','ROWS') or key in found:raise ValueError('UNEXPECTED_ASSIGNMENT')
        found[key]=ast.literal_eval(node.value)
    if set(found)!= {'HEADER','ROWS'}:raise ValueError('INCOMPLETE_CHUNK')
    h=found['HEADER'];raw=rows_to_bytes(found['ROWS'],h['dtype'],start=h['start'])
    if len(found['ROWS'])!=h['count'] or digest(raw)!=h['raw_sha256']:
        raise ValueError('CHUNK_DIGEST_MISMATCH')
    return h,raw

def export_tensor(t:torch.Tensor,out:Path,*,name:str,chunk_elements:int=4096):
    if chunk_elements<=0:raise ValueError('positive chunk size required')
    if not t.is_contiguous():raise ValueError('EXPORT_REQUIRES_CONTIGUOUS: copy explicitly, do not hide a large allocation')
    dtype=TORCH_NAMES[t.dtype]
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    flat=t.detach().view(-1)
    chunks=[];total=__import__('hashlib').sha256()
    for start in range(0,t.numel(),chunk_elements):
        raw=tensor_bytes(flat[start:start+chunk_elements]);total.update(raw)
        p=out/f'chunk_{start:012d}.py'
        h=write_chunk(p,name=name,dtype=dtype,shape=t.shape,start=start,raw=raw)
        chunks.append({'file':p.name,'file_sha256':digest(p.read_bytes()),**h})
    manifest={'format':'XMO-READABLE-TENSOR-1','name':name,'dtype':dtype,
              'shape':list(t.shape),'chunks':chunks,'raw_sha256':total.hexdigest(),
              'original_device':str(t.device),'elements':t.numel()}
    new_file(out/'MANIFEST.json',canonical(manifest))
    return manifest

def restore_tensor(out:Path,*,device='cpu'):
    import json,hashlib
    out=Path(out);m=json.loads((out/'MANIFEST.json').read_text())
    # This in-memory reconstruction is for a selected tensor, not all 9B at once.
    raw=bytearray();expected=0
    for row in m['chunks']:
        p=out/row['file']
        if p.parent!=out or p.is_symlink() or digest(p.read_bytes())!=row['file_sha256']:
            raise ValueError('CHUNK_PATH_OR_IDENTITY_MISMATCH')
        h,b=read_chunk(p)
        if (h['name']!=m['name'] or h['dtype']!=m['dtype'] or list(h['shape'])!=m['shape'] or h['start']!=expected):
            raise ValueError('CHUNK_TENSOR_MISMATCH')
        raw.extend(b);expected+=h['count']
    if expected!=m['elements'] or hashlib.sha256(raw).hexdigest()!=m['raw_sha256']:
        raise ValueError('TENSOR_DIGEST_MISMATCH')
    return tensor_from_bytes(bytes(raw),m['dtype'],m['shape'],device=device)
