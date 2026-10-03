"""Append-only evidence with checked writes, hash chaining, and an explicit seal.

The journal authenticates internal consistency, NOT who produced the evidence.
Retain the seal hash independently to detect a wholesale rewrite/reseal.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

class EvidenceError(RuntimeError):
    """Recording or verification failed; never interpret the run as complete."""

def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def checked_write(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError('write returned no progress')
        view = view[n:]

def new_file(path: Path, data: bytes) -> None:
    # Windows text-mode descriptors translate LF, invalidating byte hashes.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    try:
        checked_write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)

class Journal:
    """One writer process, serialized threads; no sampling, rollover, or event cap.

    A fresh directory is mandatory. fsync_each=True is deliberately slow but
    exposes write/fsync errors at the emitting operation. A hard process crash
    yields an unsealed run even if all preceding JSON lines remain readable.
    """
    def __init__(self, root: str | Path, *, run_id: str, fsync_each: bool = True,
                 request_id: str | None = None, phase: str = "diagnostic",
                 producer_id: str | None = None):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('NONEMPTY_RUN_ID_REQUIRED')
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError('INVALID_REQUEST_ID')
        if not isinstance(phase, str) or not phase:
            raise ValueError('NONEMPTY_PHASE_REQUIRED')
        if producer_id is not None and (not isinstance(producer_id, str) or not producer_id):
            raise ValueError('INVALID_PRODUCER_ID')
        self.run_id, self.request_id, self.phase = run_id, request_id, phase
        self.producer_id = producer_id or uuid.uuid4().hex
        self.thread_sequences = {}
        self.call_events = {}
        self.call_sources = {}
        self.last_flushed_sequence = -1
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.blob_root = self.root / 'blobs'
        self.blob_root.mkdir()
        self.owner_pid = os.getpid()
        self.lock = threading.RLock()
        self.seq = 0
        self.tail = '0' * 64
        self.closed = False
        self.failed: str | None = None
        self.fsync_each = fsync_each
        self.blobs: dict[str, int] = {}
        # Bounded derived summaries of immutable, content-addressed blobs.
        # No tensor/storage references and no version-counter-based reuse.
        self._blob_summaries = OrderedDict()
        self._blob_summary_limit = 4096
        self.artifacts: dict[str, dict] = {}
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
        self.fd = os.open(self.root / 'events.jsonl', flags, 0o600)
        self.emit('RUN_START', run_id=run_id, writer_pid=self.owner_pid,
                  fsync_each=fsync_each, event_sampling=False,
                  security='HASH_CHAIN_NOT_DIGITAL_SIGNATURE')

    def _check(self):
        if os.getpid() != self.owner_pid:
            raise EvidenceError('FORKED_WRITER: open a separate journal for each process')
        if self.closed:
            raise EvidenceError('JOURNAL_CLOSED')
        if self.failed:
            raise EvidenceError('JOURNAL_ALREADY_FAILED: ' + self.failed)

    def _fail(self, exc: BaseException):
        self.failed = f'{type(exc).__name__}: {exc}'
        # The manifest must never be used to conceal a write error. This signal
        # is best effort: a full/unwritable disk can prevent even this write.
        try:
            new_file(self.root / 'INCOMPLETE.json', canonical({'reason': self.failed,
                'last_successful_seq': self.seq - 1, 'tail': self.tail}))
        except OSError:
            pass

    def emit(self, kind: str, **fields) -> int:
        with self.lock:
            self._check()
            tid = threading.get_ident()
            cid = fields.get('call_id')
            is_enter = kind.endswith('_ENTER') or kind == 'PYTHON_CALL'
            is_terminal = kind.endswith(('_RETURN', '_ERROR')) and kind != 'PYTHON_RETURN'
            parent_call = fields.get('parent_call_id')
            if parent_call is None and fields.get('parent_scope'):
                parent_call = fields['parent_scope'][-1]
            parent = self.call_events.get(parent_call) if is_enter else self.call_events.get(cid)
            source = fields.get('source')
            source = source if isinstance(source, dict) else {}
            source_hash = (fields.get('source_sha256') or source.get('source_file_sha256')
                           or self.call_sources.get(cid))
            now = time.monotonic_ns()
            row = {'seq': self.seq, 'kind': kind, 'pid': self.owner_pid,
                   'thread_id': tid, 'wall_ns': time.time_ns(),
                   'monotonic_ns': now, 'prev_sha256': self.tail,
                   'schema_version': 'xiaomai.trace.event.v1',
                   'run_id': self.run_id, 'request_id': self.request_id, 'phase': self.phase,
                   'event_type': kind, 'process_id': self.owner_pid,
                   'sequence': self.seq, 'monotonic_time': now,
                   'monotonic_time_unit': 'nanoseconds',
                   'producer_id': self.producer_id,
                   'producer_sequence': self.seq,
                   'thread_sequence': self.thread_sequences.get(tid, 0),
                   'event_id': self.producer_id + ':' + str(self.seq),
                   'source_id': 'sha256:' + source_hash if source_hash else None,
                   'parent_event_id': parent,
                   'null_reasons': {
                       **({'request_id': 'NOT_PROVIDED_BY_CALLER'} if self.request_id is None else {}),
                       **({'source_id': 'NO_SOURCE_ID_RESOLVED_AT_THIS_BOUNDARY'} if not source_hash else {}),
                       **({'parent_event_id': 'NO_RECORDED_PARENT_AT_THIS_BOUNDARY'} if parent is None else {})},
                   'data': fields}
            try:
                h = digest(canonical(row))
                checked_write(self.fd, canonical({**row, 'sha256': h}) + b'\n')
                if self.fsync_each:
                    os.fsync(self.fd)
            except BaseException as exc:
                self._fail(exc)
                raise EvidenceError('EVENT_WRITE_FAILED') from exc
            self.thread_sequences[tid] = row['thread_sequence'] + 1
            if is_enter and cid is not None:
                self.call_events[cid] = row['event_id']
                self.call_sources[cid] = source_hash
            if is_terminal and cid is not None:
                self.call_events.pop(cid, None)
                self.call_sources.pop(cid, None)
            if kind == 'PYTHON_RETURN' and fields.get('opcode_at_event') not in ('YIELD_VALUE', 'YIELD_FROM'):
                self.call_events.pop(cid, None)
                self.call_sources.pop(cid, None)
            if self.fsync_each:
                self.last_flushed_sequence = self.seq
            previous = self.seq
            self.seq += 1
            self.tail = h
            return previous

    def blob(self, data: bytes) -> dict:
        with self.lock:
            self._check()
            h = digest(data)
            if h not in self.blobs:
                try:
                    new_file(self.blob_root / h, data)
                except BaseException as exc:
                    self._fail(exc)
                    raise EvidenceError('BLOB_WRITE_FAILED') from exc
                self.blobs[h] = len(data)
            return {'sha256': h, 'bytes': len(data), 'path': 'blobs/' + h}

    def cached_blob_summary(self, reference: dict, namespace: str, build) -> tuple[dict, bool]:
        """Reuse a flat derived summary only for already authenticated content.

        Call blob(current_bytes) at every observation boundary first. The cache
        never excuses reading/hashing current tensor bytes; it only avoids
        rescanning the same immutable content for derived numeric statistics.
        Namespaces must encode dtype and summary algorithm, not tensor identity.
        Eviction affects performance only and does not delete evidence.
        """
        with self.lock:
            self._check()
            h = reference.get('sha256')
            if (type(namespace) is not str or not namespace or
                    h not in self.blobs or reference.get('bytes') != self.blobs[h] or
                    reference.get('path') != 'blobs/' + h):
                raise EvidenceError('SUMMARY_REQUIRES_REGISTERED_CONTENT_BLOB')
            key = (namespace, h)
            if key in self._blob_summaries:
                self._blob_summaries.move_to_end(key)
                return dict(self._blob_summaries[key]), True
            summary = dict(build())
            self._blob_summaries[key] = summary
            if len(self._blob_summaries) > self._blob_summary_limit:
                self._blob_summaries.popitem(last=False)
            return dict(summary), False

    def artifact(self, relative_path: str) -> dict:
        with self.lock:
            self._check()
            p = self.root / relative_path
            if p.is_symlink() or p.parent != self.root or relative_path in ('events.jsonl','SEAL.json','INCOMPLETE.json'):
                raise EvidenceError('UNSAFE_ARTIFACT_PATH')
            h = hashlib.sha256(); size = 0
            try:
                with p.open('rb') as f:
                    while block := f.read(1 << 20):
                        h.update(block); size += len(block)
            except OSError as exc:
                self._fail(exc); raise EvidenceError('ARTIFACT_READ_FAILED') from exc
            ref = {'kind':'artifact_reference','path':relative_path,'sha256':h.hexdigest(),'bytes':size}
            self.artifacts[relative_path] = ref
            return ref

    def flush(self) -> dict:
        with self.lock:
            self._check()
            try:
                os.fsync(self.fd)
            except BaseException as exc:
                self._fail(exc)
                raise EvidenceError('FLUSH_FAILED') from exc
            self.last_flushed_sequence = self.seq - 1
            return {'producer_id': self.producer_id, 'through_sequence': self.last_flushed_sequence,
                    'tail_sha256': self.tail, 'fsync_completed': True}

    def seal(self, *, execution: str, coverage: dict) -> dict:
        with self.lock:
            self._check()
            self.emit('RUN_END', execution=execution, coverage=coverage)
            try:
                os.fsync(self.fd)
                value = {'format': 'XMO-JOURNAL-1', 'events': self.seq,
                         'tail_sha256': self.tail, 'execution': execution,
                         'coverage': coverage, 'blobs': self.blobs, 'artifacts': self.artifacts,
                         'physical_power_loss_durability_verified': False,
                         'identity': {'run_id': self.run_id, 'request_id': self.request_id,
                                      'phase': self.phase, 'producer_id': self.producer_id,
                                      'process_id': self.owner_pid},
                         'flushed_through_sequence': self.seq - 1}
                new_file(self.root / 'SEAL.json', canonical(value))
            except BaseException as exc:
                self._fail(exc)
                raise EvidenceError('SEAL_FAILED') from exc
            finally:
                os.close(self.fd)
                self.closed = True
            return value

    def abort(self) -> None:
        with self.lock:
            if not self.closed:
                self._fail(EvidenceError('ABORTED_WITHOUT_SEAL'))
                os.close(self.fd)
                self.closed = True


def verify(root: str | Path, *, expected_seal_sha256: str | None = None,
           tick=None, file_digest=None) -> dict:
    if tick:tick()
    root = Path(root)
    if (root / 'INCOMPLETE.json').exists():
        raise EvidenceError('RUN_MARKED_INCOMPLETE')
    try:
        raw_seal = (root / 'SEAL.json').read_bytes()
        if tick:tick()
        seal = json.loads(raw_seal)
        if expected_seal_sha256 and digest(raw_seal) != expected_seal_sha256:
            raise EvidenceError('SEAL_IDENTITY_MISMATCH')
        tail, count, last = '0' * 64, 0, None
        referenced_blobs = {}
        referenced_artifacts = {}
        open_calls = {}
        def check_refs(value):
            if isinstance(value, dict):
                if value.get('kind')=='artifact_reference':
                    referenced_artifacts[value['path']]=value
                if set(('sha256','bytes','path')) <= value.keys() and str(value['path']).startswith('blobs/'):
                    if value['path'] != 'blobs/' + value['sha256']:
                        raise EvidenceError('INVALID_BLOB_PATH')
                    referenced_blobs[value['sha256']] = value['bytes']
                for v in value.values(): check_refs(v)
            elif isinstance(value, list):
                for v in value: check_refs(v)
        with (root / 'events.jsonl').open('rb') as inp:
            for line in inp:
                if tick:tick()
                if not line.endswith(b'\n'):
                    raise EvidenceError('TRUNCATED_EVENT')
                row = json.loads(line)
                h = row.pop('sha256')
                if row['seq'] != count or row['prev_sha256'] != tail or digest(canonical(row)) != h:
                    raise EvidenceError(f'CHAIN_MISMATCH_AT_{count}')
                check_refs(row)
                kind=row['kind']; fields=row['data']
                if kind in ('MODULE_ENTER','ATEN_ENTER'):
                    cid=fields['call_id']
                    if cid in open_calls:raise EvidenceError('DUPLICATE_CALL_ID')
                    parent=fields.get('parent_call_id')
                    if parent is not None and parent not in open_calls:raise EvidenceError('UNKNOWN_PARENT_CALL')
                    open_calls[cid]=kind.split('_')[0]
                elif kind in ('MODULE_RETURN','MODULE_ERROR','ATEN_RETURN','ATEN_ERROR'):
                    cid=fields['call_id']
                    if open_calls.pop(cid,None)!=kind.split('_')[0]:raise EvidenceError('UNMATCHED_CALL_TERMINAL')
                tail, count, last = h, count + 1, row
        if open_calls:raise EvidenceError('UNCLOSED_INSTRUMENTED_CALLS')
        if referenced_artifacts!=seal.get('artifacts',{}):raise EvidenceError('ARTIFACT_SET_MISMATCH')
        for name,ref in referenced_artifacts.items():
            p=root/name
            if p.parent!=root or p.is_symlink():raise EvidenceError('UNSAFE_ARTIFACT_PATH')
            if tick:tick()
            if file_digest is not None:
                measured=file_digest(p,expected_sha256=ref['sha256'],expected_bytes=ref['bytes'],tick=tick)
                if measured['bytes']!=ref['bytes'] or measured['sha256']!=ref['sha256']:
                    raise EvidenceError('ARTIFACT_MISMATCH')
            else:
                h=hashlib.sha256();size=0
                with p.open('rb') as f:
                    while b:=f.read(1<<20):
                        h.update(b);size+=len(b)
                        if tick:tick()
                if size!=ref['bytes'] or h.hexdigest()!=ref['sha256']:raise EvidenceError('ARTIFACT_MISMATCH')
        if count != seal['events'] or tail != seal['tail_sha256']:
            raise EvidenceError('SEAL_TAIL_MISMATCH')
        if last is None or last['kind'] != 'RUN_END':
            raise EvidenceError('MISSING_TERMINAL_EVENT')
        if last['data']['execution'] != seal['execution'] or last['data']['coverage'] != seal['coverage']:
            raise EvidenceError('SEAL_EVENT_DISAGREEMENT')
        for h, n in referenced_blobs.items():
            if seal['blobs'].get(h) != n:
                raise EvidenceError('UNSEALED_BLOB_REFERENCE')
        actual = set()
        for p in (root/'blobs').iterdir():
            if tick:tick()
            if p.is_file():actual.add(p.name)
        if actual != set(seal['blobs']):
            raise EvidenceError('BLOB_SET_MISMATCH')
        for h,n in seal['blobs'].items():
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise EvidenceError('BAD_BLOB_DIGEST')
            if tick:tick()
            if file_digest is not None:
                measured=file_digest(root/'blobs'/h,expected_sha256=h,expected_bytes=n,tick=tick)
                if measured['bytes']!=n or measured['sha256']!=h:
                    raise EvidenceError('BLOB_MISMATCH')
            else:
                if tick is None:
                    b=(root/'blobs'/h).read_bytes()
                    size,found=len(b),digest(b)
                else:
                    hasher=hashlib.sha256();size=0
                    with (root/'blobs'/h).open('rb') as inp:
                        while block:=inp.read(1<<20):
                            hasher.update(block);size+=len(block);tick()
                    found=hasher.hexdigest()
                if size!=n or found!=h:raise EvidenceError('BLOB_MISMATCH')
        if tick:tick()
        return {'status':'CONSISTENT_SEALED_EVIDENCE', 'events':count,
                'execution':seal['execution'], 'coverage':seal['coverage'],
                'seal_sha256':digest(raw_seal), 'blobs':len(seal['blobs'])}
    except EvidenceError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceError('MALFORMED_OR_MISSING_EVIDENCE') from exc
