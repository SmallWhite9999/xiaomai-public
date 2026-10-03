"""Losslessly compressed full event stream, not sampled trace summaries.

Each event has an ordinal and hash-chain link. The compressed file is sealed
only after GZIP close, fsync and independent readback. An independently retained
seal hash authenticates that version; the hash chain is NOT a signature.
"""
from __future__ import annotations
import gzip,hashlib,json,os,threading
from pathlib import Path

class TraceError(RuntimeError):pass

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        while raw:=stream.read(1<<20):h.update(raw)
    return h.hexdigest()

class CompressedTrace:
    def __init__(self,root,*,run_id,phase):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.run_id=run_id;self.phase=phase;self.pid=os.getpid();self.lock=threading.RLock()
        self.file=(self.root/'events.jsonl.gz').open('xb')
        self.writer=gzip.GzipFile(filename='',mode='wb',fileobj=self.file,mtime=0,compresslevel=6)
        self.count=0;self.tail='0'*64;self.bytes=0;self.failed=False;self.closed=False;self.receipt=None
        self.emit('TRACE_BEGIN',scope='all emitted events, no sampling',durability='flush/fsync on phase close',
                  compression='gzip, reversible',not_a_digital_signature=True)
    def emit(self,kind,**fields):
        with self.lock:
            if self.pid!=os.getpid():raise TraceError('FORKED_WRITER_REJECTED')
            if self.closed or self.failed:raise TraceError('TRACE_NOT_WRITABLE')
            row={'seq':self.count,'previous_sha256':self.tail,'kind':kind,'run_id':self.run_id,
                 'phase':self.phase,'pid':self.pid,'thread':threading.get_ident(),'data':fields}
            try:
                h=digest(canonical(row));raw=canonical({**row,'sha256':h})+b'\n'
                n=self.writer.write(raw)
                if n!=len(raw):raise TraceError('SHORT_WRITE')
                self.bytes+=len(raw);self.tail=h;self.count+=1
            except BaseException:
                self.failed=True;raise
    def close(self,*,execution):
        if self.closed:raise TraceError('ALREADY_CLOSED')
        try:
            if not self.failed:self.emit('TRACE_END',execution=execution)
            self.writer.close();self.file.flush();os.fsync(self.file.fileno());self.file.close();self.closed=True
            if self.failed or execution!='RETURNED':return None
            seal={'schema':'XIAOMAI_COMPRESSED_TRACE_V1','execution':execution,'events':self.count,
                  'tail_sha256':self.tail,'uncompressed_bytes':self.bytes,
                  'compressed_bytes':(self.root/'events.jsonl.gz').stat().st_size,
                  'compressed_sha256':file_hash(self.root/'events.jsonl.gz')}
            # Only finalize a stream that can be independently decoded and counted.
            _verify_stream(self.root/'events.jsonl.gz',seal)
            raw=canonical(seal)
            with (self.root/'SEAL.json').open('xb') as f:
                if f.write(raw)!=len(raw):raise TraceError('SEAL_SHORT_WRITE')
                f.flush();os.fsync(f.fileno())
            self.receipt={**seal,'seal_sha256':digest(raw)};return self.receipt
        except BaseException:
            self.failed=True
            if not self.closed:
                try:self.writer.close()
                finally:self.file.close();self.closed=True
            raise
    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):self.close(execution='RETURNED' if exc_type is None else 'RAISED')

def _verify_stream(path,seal):
    tail='0'*64;count=0;size=0;end=None;kinds={};phases=set();pids=set()
    try:
        with gzip.open(path,'rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):raise TraceError('TRUNCATED_LINE')
                row=json.loads(line);h=row.pop('sha256')
                if row['seq']!=count or row['previous_sha256']!=tail or digest(canonical(row))!=h:raise TraceError('EVENT_CHAIN_MISMATCH')
                if count==0 and row['kind']!='TRACE_BEGIN':raise TraceError('MISSING_BEGIN')
                if end is not None:raise TraceError('EVENT_AFTER_END')
                if row['kind']=='TRACE_END':end=row['data']['execution']
                kinds[row['kind']]=kinds.get(row['kind'],0)+1;phases.add(row['phase']);pids.add(row['pid'])
                tail=h;count+=1;size+=len(line)
    except (OSError,EOFError,json.JSONDecodeError,KeyError,TypeError) as exc:raise TraceError('COMPRESSED_TRACE_INVALID') from exc
    if count!=seal['events'] or size!=seal['uncompressed_bytes'] or tail!=seal['tail_sha256'] or end!='RETURNED':raise TraceError('UNSEALED_OR_INCOMPLETE_EVENTS')
    if len(phases)!=1 or len(pids)!=1:raise TraceError('PHASE_OR_PID_MIXED')
    return {'events':count,'kinds':kinds,'phases':sorted(phases),'pids':sorted(pids),'bytes':size}

def verify(root,*,expected_seal_sha256):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise TraceError('TRACE_DIRECTORY')
    if {p.name for p in root.iterdir()}!={'events.jsonl.gz','SEAL.json'}:raise TraceError('TRACE_FILE_SET')
    if any(p.is_symlink() for p in root.iterdir()):raise TraceError('SYMLINK_REJECTED')
    raw=(root/'SEAL.json').read_bytes()
    if digest(raw)!=expected_seal_sha256:raise TraceError('EXTERNAL_SEAL_MISMATCH')
    seal=json.loads(raw)
    if seal.get('schema')!='XIAOMAI_COMPRESSED_TRACE_V1' or seal.get('execution')!='RETURNED':raise TraceError('BAD_SEAL')
    p=root/'events.jsonl.gz'
    if p.stat().st_size!=seal['compressed_bytes'] or file_hash(p)!=seal['compressed_sha256']:raise TraceError('COMPRESSED_FILE_MISMATCH')
    return {'status':'LOSSLESS_TRACE_VERIFIED',**_verify_stream(p,seal),'compressed_bytes':p.stat().st_size}
