"""Bounded digest of generated evidence, separate from pinned model assets.

Windows can hash once while a native handle excludes write/delete sharing.
POSIX has no equivalent global exclusion here: retain the original double-pass
checked_digest contract, including its authenticated block recheck. Never claim
that a stat comparison alone proves unchanging POSIX content.
"""
from __future__ import annotations
import hashlib
import os
import stat

from .long_paths import no_links
from .real_assets import checked_digest
from .source_handles import AssetError, _open_readonly, stat_identity, BLOCK_BYTES


def generated_evidence_digest(path, *, expected_sha256=None, expected_bytes=None,
                              tick=None, block_bytes=BLOCK_BYTES):
    """Return a digest only after link, identity, length and optional pin checks.

    No returned reader or data buffer can outlive the held handle. ``tick`` runs
    before every bounded read, including the final EOF check; exceptions abort.
    No callback runs after the final identity/content checks. The source-asset
    admission and consumed-block verification implementation remains unchanged.
    """
    if type(block_bytes) is not int or block_bytes<=0:
        raise ValueError('POSITIVE_BLOCK_SIZE_REQUIRED')
    if os.name!='nt':
        return {**checked_digest(path,expected_sha256=expected_sha256,
                    expected_bytes=expected_bytes,tick=tick),
                'digest_passes':2,'digest_purpose':'GENERATED_EVIDENCE'}
    path=no_links(path)
    with _open_readonly(path) as stream:
        admitted=os.fstat(stream.fileno())
        if not stat.S_ISREG(admitted.st_mode):
            raise AssetError('REGULAR_EVIDENCE_FILE_REQUIRED')
        identity=stat_identity(admitted)
        if stat_identity(path.stat())!=identity:
            raise AssetError('EVIDENCE_PATH_HANDLE_MISMATCH:'+path.name)
        if expected_bytes is not None and identity[2]!=expected_bytes:
            raise AssetError('SIZE_MISMATCH:'+path.name)
        total=0;blocks=0;whole=hashlib.sha256()
        while True:
            remaining=identity[2]-total
            # Raw file reads allocate for the requested size even for tiny files
            # and EOF. Bound allocation to admitted remaining content, but still
            # read one byte after it to reject an unexpected trailing byte.
            request=min(block_bytes,remaining) if remaining else 1
            if tick:tick()
            raw=stream.read(request)
            if not raw:break
            whole.update(raw);total+=len(raw);blocks+=1
            if total>identity[2]:
                raise AssetError('EVIDENCE_LENGTH_CHANGED_DURING_READ:'+path.name)
        no_links(path)
        if (stat_identity(os.fstat(stream.fileno()))!=identity
                or stat_identity(path.stat())!=identity):
            raise AssetError('EVIDENCE_CHANGED_DURING_READ:'+path.name)
        if total!=identity[2]:
            raise AssetError('EVIDENCE_LENGTH_CHANGED_DURING_READ:'+path.name)
        digest=whole.hexdigest()
        if expected_sha256 is not None and digest!=expected_sha256:
            raise AssetError('SHA256_MISMATCH:'+path.name)
        return {'bytes':total,'sha256':digest,'stat':list(identity),
                'read_policy':'WINDOWS_HELD_HANDLE_DENY_WRITE_DELETE_SINGLE_PASS',
                'block_bytes':block_bytes,'blocks':blocks,'digest_passes':1,
                'digest_purpose':'GENERATED_EVIDENCE',
                'posix_global_write_exclusion_claimed':False}
