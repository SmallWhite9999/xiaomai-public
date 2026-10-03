"""Lossless content-addressed frame-local snapshots and boundary transitions.

No sampling: every eligible boundary gets a snapshot reference. Equal snapshots
share bytes; tensors are already immutable blobs in Capture. This does not read
CPython's private operand stack or infer a semantic cause from a value change.
"""
from __future__ import annotations
import json
from pathlib import Path
from .journal import canonical, digest, EvidenceError

class LocalSnapshots:
    def __init__(self, journal):
        self.journal=journal;self.references=0;self.unique=set();self.logical_bytes=0;self.stored_bytes=0

    def put(self, safe_value):
        raw=canonical(safe_value);self.references+=1;self.logical_bytes+=len(raw)
        ref=self.journal.blob(raw)
        if ref['sha256'] not in self.unique:
            self.unique.add(ref['sha256']);self.stored_bytes+=len(raw)
        return {'kind':'python_locals_snapshot','format':'XMO-LOCALS-1',**ref}

    def stats(self):
        return {'references':self.references,'unique_snapshots':len(self.unique),
                'logical_snapshot_bytes':self.logical_bytes,'unique_snapshot_bytes':self.stored_bytes,
                'dropped_boundaries':0,'numeric_sampling':False}


def load_snapshot(root: Path, ref: dict):
    if ref.get('format')!='XMO-LOCALS-1':raise EvidenceError('LOCAL_SNAPSHOT_FORMAT')
    h=ref['sha256']
    if len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or ref['path']!='blobs/'+h:
        raise EvidenceError('LOCAL_SNAPSHOT_PATH')
    root=Path(root);p=root/'blobs'/h
    if p.is_symlink() or (root/'blobs').is_symlink():raise EvidenceError('LOCAL_SNAPSHOT_SYMLINK')
    raw=p.read_bytes()
    if len(raw)!=ref['bytes'] or digest(raw)!=h:raise EvidenceError('LOCAL_SNAPSHOT_HASH')
    return json.loads(raw)


def variable_fingerprints(safe_value):
    if not isinstance(safe_value,dict) or safe_value.get('kind')!='mapping':return {}
    # Frame-local names are ordinary strings. Do not coerce user mapping keys.
    return {k:digest(canonical(v)) for k,v in safe_value['items'] if type(k) is str}


def diff_names(before: dict, after: dict):
    return {'added':sorted(after.keys()-before.keys()),'removed':sorted(before.keys()-after.keys()),
            'changed':sorted(k for k in before.keys()&after.keys() if before[k]!=after[k])}
