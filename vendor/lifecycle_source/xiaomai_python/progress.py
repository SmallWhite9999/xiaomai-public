"""Durable stage/progress events. Progress is not a substitute for full trace."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from .long_paths import mkdir_safe

class Progress:
    def __init__(self, path, *, spec_sha256=None, budget=None, producer=None):
        self.path = Path(path)
        mkdir_safe(self.path.parent)
        self.file = self.path.open('xb')
        self.spec_sha256 = spec_sha256
        self.budget = budget
        self.producer = producer or uuid.uuid4().hex
        self.pid = os.getpid()
        self.seq = 0
        self.previous = '0'*64
        self.lock = threading.RLock()
        self.last_work_tick = 0.0
        self.current_phase = None
        self.closed = False

    def emit(self, event, **fields):
        with self.lock:
            row = {'schema':'xiaomai.progress.v082', 'event':event, 'seq':self.seq,
                   'producer':self.producer, 'pid':self.pid, 'thread':threading.get_ident(),
                   'monotonic_ns':time.monotonic_ns(), 'time_ns':time.time_ns(),
                   'run_spec_sha256':self.spec_sha256,'previous_sha256':self.previous, **fields}
            raw = json.dumps(row, sort_keys=True, separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
            digest = hashlib.sha256(raw).hexdigest()
            row['event_sha256'] = digest
            data = json.dumps(row, sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8') + b'\n'
            self.file.write(data)
            self.file.flush()
            os.fsync(self.file.fileno())
            self.previous = digest
            self.seq += 1
            return row

    @contextlib.contextmanager
    def phase(self, name):
        previous = self.current_phase
        self.current_phase = name
        start = time.monotonic()
        self.emit('PHASE_BEGIN',phase=name)
        try:
            if self.budget: self.budget.check(force=True)
            yield
            if self.budget: self.budget.check(force=True)
        except BaseException as e:
            self.emit('PHASE_ERROR',phase=name,seconds=time.monotonic()-start,error_type=type(e).__name__)
            raise
        else:
            self.emit('PHASE_END',phase=name,seconds=time.monotonic()-start)
        finally:
            self.current_phase = previous

    def asset(self, row):
        self.emit('ASSET_PROGRESS',phase=self.current_phase,detail=row)

    def operator(self, event, **fields):
        self.emit(event,phase=self.current_phase,**fields)
        if self.budget: self.budget.check()

    def work(self, **fields):
        """Cooperative tile boundary check; heartbeat cadence is explicitly 1s.
        This stream is progress only. The numerical trace is independently
        recorded at its selected scope and is never sampled by this function.
        """
        if self.budget: self.budget.check()
        now = time.monotonic()
        if now - self.last_work_tick >= 1.0:
            self.last_work_tick = now
            self.emit('WORK_PROGRESS',phase=self.current_phase,heartbeat_seconds=1.0,**fields)

    def close(self):
        if not self.closed:
            self.closed = True
            self.file.close()

    def __enter__(self): return self
    def __exit__(self,*a): self.close()


def verify_progress(path, *, spec_sha256=None, required_phases=(), tick=None):
    previous = '0'*64
    ended = []
    active = []
    count = 0
    with Path(path).open('rb') as f:
        for line in f:
            if tick:tick()
            if not line.endswith(b'\n'): raise ValueError('PROGRESS_TRUNCATED')
            row = json.loads(line)
            digest = row.pop('event_sha256')
            raw = json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
            if hashlib.sha256(raw).hexdigest()!=digest or row['seq']!=count or row['previous_sha256']!=previous:
                raise ValueError('PROGRESS_CHAIN_MISMATCH')
            if row['schema']!='xiaomai.progress.v082' or row['run_spec_sha256']!=spec_sha256:
                raise ValueError('PROGRESS_IDENTITY_MISMATCH')
            if row['event']=='PHASE_BEGIN': active.append(row['phase'])
            if row['event']=='PHASE_ERROR': raise ValueError('PROGRESS_HAS_ERROR')
            if row['event']=='PHASE_END':
                if not active or active.pop()!=row['phase']: raise ValueError('PROGRESS_UNPAIRED_PHASE')
                ended.append(row['phase'])
            previous = digest; count += 1
    if not count or active or any(p not in ended for p in required_phases):
        raise ValueError('PROGRESS_MISSING_PHASE')
    if tick:tick()
    return {'events':count,'last_sha256':previous,'phases':ended}
