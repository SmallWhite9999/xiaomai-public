"""Full logical event recording with lossless block compression.

Every event and its sequence are stored. Hashes bind blocks plus their ordered
index, avoiding two random SHA strings at every line. No sampling, no rotation,
no omission of an emitted event. A crash produces no complete seal.
"""
from __future__ import annotations
import gzip,hashlib,json,os,stat,threading
from pathlib import Path
from .compressed_trace import canonical,digest,TraceError

def _link_or_reparse(path):
    """Reject Windows junctions as well as symbolic links before reading."""
    info=path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400))

def _new(path,raw):
    with path.open('xb') as f:
        if f.write(raw)!=len(raw):raise TraceError('SHORT_WRITE')
        f.flush();os.fsync(f.fileno())

class BlockTrace:
    def __init__(self,root,*,run_id,phase,block_events=1024,block_bytes=1048576):
        if type(block_events) is not int or not 1<=block_events<=65536:raise TraceError('BLOCK_EVENT_LIMIT')
        if type(block_bytes) is not int or not 1024<=block_bytes<=16*1024**2:raise TraceError('BLOCK_BYTE_LIMIT')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.pid=os.getpid();self.run_id=run_id;self.phase=phase;self.block_events=block_events;self.block_bytes=block_bytes
        self.lock=threading.RLock();self.count=0;self.uncompressed=0;self.pending=bytearray();self.pending_count=0
        self.blocks=[];self.previous='0'*64;self.closed=False;self.failed=False;self.receipt=None
        self.logical_hash=hashlib.sha256()
        self.emit('TRACE_BEGIN',event_sampling=False,format='lossless logical events with ordered block hashes')
    def emit(self,kind,**data):
        with self.lock:
            if self.closed or self.failed or os.getpid()!=self.pid:raise TraceError('WRITER_STATE_OR_PID')
            row={'seq':self.count,'kind':kind,'run_id':self.run_id,'phase':self.phase,
                 'pid':self.pid,'thread':threading.get_ident(),'data':data}
            try:
                raw=canonical(row)+b'\n'
                if len(raw)>16*1024**2:raise TraceError('SINGLE_EVENT_TOO_LARGE_NO_TRUNCATION')
                if self.pending and len(raw)+len(self.pending)>self.block_bytes:self._flush()
                self.pending.extend(raw);self.pending_count+=1;self.count+=1
                self.logical_hash.update(raw);self.uncompressed+=len(raw)
                if self.pending_count>=self.block_events or len(self.pending)>=self.block_bytes:self._flush()
            except BaseException:self.failed=True;raise
    def _flush(self):
        if not self.pending:return
        ordinal=len(self.blocks);name=f'block-{ordinal:08d}.jsonl.gz';raw=bytes(self.pending)
        compressed=gzip.compress(raw,compresslevel=6,mtime=0)
        record={'file':name,'first_seq':self.count-self.pending_count,'events':self.pending_count,
                'uncompressed_bytes':len(raw),'uncompressed_sha256':digest(raw),
                'compressed_bytes':len(compressed),'compressed_sha256':digest(compressed),
                'previous_block_sha256':self.previous}
        _new(self.root/name,compressed);encoded=canonical(record)
        _new(self.root/f'block-{ordinal:08d}.json',encoded)
        self.previous=digest(encoded);self.blocks.append(record);self.pending.clear();self.pending_count=0
    def close(self,*,execution):
        with self.lock:
            if self.closed:raise TraceError('ALREADY_CLOSED')
            try:
                if self.failed:self.closed=True;return None
                self.emit('TRACE_END',execution=execution);self._flush();self.closed=True
                if execution!='RETURNED':return None
                seal={'schema':'XIAOMAI_LOSSLESS_BLOCK_TRACE_V1','status':'COMPLETE','execution':execution,
                      'events':self.count,'blocks':self.blocks,'logical_sha256':self.logical_hash.hexdigest(),
                      'uncompressed_bytes':self.uncompressed,'last_block_sha256':self.previous}
                encoded=canonical(seal)
                # _verify validates every row before publishing the seal.
                _verify(self.root,seal,expect_seal=False)
                _new(self.root/'SEAL.json',encoded)
                self.receipt={'seal_sha256':digest(encoded),'events':self.count,'uncompressed_bytes':self.uncompressed,
                              'compressed_bytes':sum(b['compressed_bytes'] for b in self.blocks)}
                return self.receipt
            except BaseException:self.failed=True;self.closed=True;raise
    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):self.close(execution='RETURNED' if exc_type is None else 'RAISED')

def _verify(root,seal,expect_seal=True):
    if seal.get('execution')!='RETURNED':raise TraceError('EXECUTION_FAILED')
    if any(_link_or_reparse(p) or not p.is_file() for p in root.iterdir()):raise TraceError('TRACE_FILE_KIND')
    count=0;total=0;prior='0'*64;logical=hashlib.sha256();kinds={};ended=False;pid=None;phase=None;run=None
    expected={'SEAL.json'} if expect_seal else set()
    for i,block in enumerate(seal['blocks']):
        name=f'block-{i:08d}.jsonl.gz';indexname=f'block-{i:08d}.json';expected|={name,indexname}
        if block['file']!=name or block['first_seq']!=count or block['previous_block_sha256']!=prior:raise TraceError('BLOCK_ORDER')
        if type(block['events']) is not int or not 0<block['events']<=65536:raise TraceError('BLOCK_COUNT')
        meta=(root/indexname).read_bytes()
        if canonical(block)!=meta:raise TraceError('BLOCK_INDEX_CHANGED')
        prior=digest(meta);compressed=(root/name).read_bytes()
        if len(compressed)!=block['compressed_bytes'] or digest(compressed)!=block['compressed_sha256']:raise TraceError('BLOCK_BYTES_CHANGED')
        if not 0<block['uncompressed_bytes']<=17*1024**2:raise TraceError('BLOCK_SIZE_BOUND')
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:raw=stream.read(block['uncompressed_bytes']+1)
        if len(raw)!=block['uncompressed_bytes'] or digest(raw)!=block['uncompressed_sha256']:raise TraceError('BLOCK_DECODE_HASH')
        lines=raw.splitlines(keepends=True)
        if len(lines)!=block['events']:raise TraceError('BLOCK_EVENT_COUNT')
        for line in lines:
            if not line.endswith(b'\n') or ended:raise TraceError('TRUNCATED_OR_AFTER_END')
            row=json.loads(line)
            if type(row.get('seq')) is not int or row['seq']!=count:raise TraceError('EVENT_SEQUENCE')
            if count==0:
                if row['kind']!='TRACE_BEGIN':raise TraceError('MISSING_BEGIN')
                pid=row['pid'];phase=row['phase'];run=row['run_id']
            if (row['pid'],row['phase'],row['run_id'])!=(pid,phase,run):raise TraceError('IDENTITY_CHANGED')
            if count>0 and row['kind']=='TRACE_BEGIN':raise TraceError('DUPLICATE_BEGIN')
            if row['kind']=='TRACE_END':
                if row['data']['execution']!='RETURNED':raise TraceError('EXECUTION_FAILED')
                ended=True
            kinds[row['kind']]=kinds.get(row['kind'],0)+1;count+=1
        total+=len(raw);logical.update(raw)
    if count!=seal['events'] or total!=seal['uncompressed_bytes'] or logical.hexdigest()!=seal['logical_sha256'] or prior!=seal['last_block_sha256'] or not ended:
        raise TraceError('INCOMPLETE_TRACE')
    if {p.name for p in root.iterdir()}!=expected or any(_link_or_reparse(p) for p in root.iterdir()):raise TraceError('TRACE_FILE_SET')
    return {'status':'ALL_LOGICAL_EVENTS_VERIFIED','events':count,'kinds':kinds,'bytes':total,
            'compressed_bytes':sum(b['compressed_bytes'] for b in seal['blocks']),
            'logical_sha256':logical.hexdigest(),'phase':phase,'pid':pid}

def verify(root,*,expected_seal_sha256):
    root=Path(root)
    if not root.is_dir() or _link_or_reparse(root):raise TraceError('DIRECTORY_REQUIRED')
    if _link_or_reparse(root/'SEAL.json'):raise TraceError('TRACE_FILE_KIND')
    raw=(root/'SEAL.json').read_bytes()
    if digest(raw)!=expected_seal_sha256:raise TraceError('EXTERNAL_SEAL_MISMATCH')
    seal=json.loads(raw)
    if seal.get('schema')!='XIAOMAI_LOSSLESS_BLOCK_TRACE_V1' or seal.get('status')!='COMPLETE':raise TraceError('SEAL_SCHEMA')
    return _verify(root,seal)

def iter_events(root,*,expected_seal_sha256):
    verify(root,expected_seal_sha256=expected_seal_sha256)
    root=Path(root);seal=json.loads((root/'SEAL.json').read_bytes())
    for b in seal['blocks']:
        compressed=(root/b['file']).read_bytes()
        if digest(compressed)!=b['compressed_sha256']:raise TraceError('BLOCK_CHANGED_DURING_READ')
        for row in gzip.decompress(compressed).splitlines():yield json.loads(row)
