"""Pinned reads through held descriptors; Windows denies write/delete sharing.

POSIX advisory locking is not advertised as immutable storage. Each consumed
block is authenticated against the admitted whole-file hash's block list.
The block list lives in this reader, not a caller-controlled receipt. Concurrent
mutations can cause rejection, but cannot silently return different tensor bytes.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import stat
import threading
from .long_paths import no_links

class AssetError(ValueError):
    pass

BLOCK_BYTES = 8 << 20


def stat_identity(s):
    # On Windows, supported Python versions can expose change time through
    # fstat().st_ctime_ns but creation time through stat().st_ctime_ns. Compare
    # the explicit creation-time field common to BOTH APIs instead. The held
    # handle still excludes writers/deletion, and consumed bytes remain hash
    # authenticated. POSIX retains ctime to detect metadata/content changes.
    stable_time = s.st_birthtime_ns if os.name == 'nt' else s.st_ctime_ns
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, stable_time)


def _open_readonly(path):
    """Use an exclusive-write-sharing Windows handle for its entire lifetime."""
    if os.name != 'nt':
        return os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb', buffering=0)
    import ctypes
    from ctypes import wintypes
    import msvcrt
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k.CreateFileW.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ ONLY; OPEN_EXISTING; OPEN_REPARSE_POINT.
    h = k.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x00200000 | 0x08000000, None)
    if h == ctypes.c_void_p(-1).value:
        raise AssetError('SOURCE_HANDLE_LOCK_FAILED:' + str(ctypes.get_last_error()))
    try:
        fd = msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        k.CloseHandle(h)
        raise
    # fd owns h now; close exactly once through the Python file object.
    return os.fdopen(fd, 'rb', buffering=0)


class PinnedReader:
    def __init__(self, path, *, expected_sha256=None, expected_bytes=None,
                 tick=None, progress=None, block_bytes=BLOCK_BYTES):
        if type(block_bytes) is not int or block_bytes <= 0:
            raise ValueError('POSITIVE_BLOCK_SIZE_REQUIRED')
        self.path = no_links(path)
        self.block_bytes = block_bytes
        self._lock = threading.RLock()
        self._file = _open_readonly(self.path)
        self.closed = False
        self.tick = tick
        self.progress = progress
        self.blocks = []
        try:
            s = os.fstat(self._file.fileno())
            if not stat.S_ISREG(s.st_mode):
                raise AssetError('REGULAR_SOURCE_FILE_REQUIRED')
            self.stat = stat_identity(s)
            self._assert_path()
            if expected_bytes is not None and s.st_size != expected_bytes:
                raise AssetError('SIZE_MISMATCH:' + self.path.name)
            whole = hashlib.sha256()
            offset = 0
            while True:
                raw = self._file.read(block_bytes)
                if not raw:
                    break
                whole.update(raw)
                self.blocks.append(hashlib.sha256(raw).hexdigest())
                offset += len(raw)
                if progress:
                    progress({'event': 'FILE_HASH_PROGRESS', 'file': self.path.name,
                              'bytes_read': offset, 'bytes_total': self.stat[2]})
                if tick:
                    tick()
            self.sha256 = whole.hexdigest()
            if offset != self.stat[2]:
                raise AssetError('SOURCE_CHANGED_DURING_READ:' + self.path.name)
            self._assert_path()
            if expected_sha256 is not None and self.sha256 != expected_sha256:
                raise AssetError('SHA256_MISMATCH:' + self.path.name)
            # Catch writes in the final tick even on filesystems with unchanged
            # or coarse stat fields. Never call a mutable tick after this pass.
            self.verify_all(tick=False)
        except BaseException:
            self.close()
            raise

    def _assert_path(self):
        if self.closed:
            raise AssetError('SOURCE_HANDLE_CLOSED')
        no_links(self.path)
        handle = stat_identity(os.fstat(self._file.fileno()))
        path = stat_identity(self.path.stat())
        if handle != self.stat or path != self.stat:
            raise AssetError('SOURCE_CHANGED_SINCE_ADMISSION:' + self.path.name)

    def _block(self, index):
        self._file.seek(index * self.block_bytes)
        raw = self._file.read(min(self.block_bytes, self.stat[2] - index*self.block_bytes))
        if hashlib.sha256(raw).hexdigest() != self.blocks[index]:
            raise AssetError('SOURCE_CHANGED_CONTENT:' + self.path.name + ':' + str(index*self.block_bytes))
        return raw

    def read(self, offset=0, size=None):
        """Return only authenticated bytes, from the same handle used at admission."""
        with self._lock:
            self._assert_path()
            if size is None:
                size = self.stat[2] - offset
            if type(offset) is not int or type(size) is not int or min(offset,size) < 0 or offset+size > self.stat[2]:
                raise AssetError('SOURCE_READ_OUT_OF_RANGE')
            output = bytearray()
            if size:
                first, last = offset//self.block_bytes, (offset+size-1)//self.block_bytes
                for i in range(first,last+1):
                    raw = self._block(i)
                    start = max(offset-i*self.block_bytes,0)
                    end = min(offset+size-i*self.block_bytes,len(raw))
                    output.extend(raw[start:end])
                    if self.tick:
                        self.tick()
            self._assert_path()
            return bytes(output)

    def verify_all(self, *, tick=True):
        with self._lock:
            self._assert_path()
            whole = hashlib.sha256()
            for i in range(len(self.blocks)):
                if tick and self.tick:
                    self.tick()
                whole.update(self._block(i))
            if whole.hexdigest() != self.sha256:
                raise AssetError('SOURCE_CHANGED_FULL_HASH:' + self.path.name)
            self._assert_path()
            return self.receipt()

    def receipt(self):
        return {'bytes': self.stat[2], 'sha256': self.sha256, 'stat': list(self.stat),
                'read_policy': 'WINDOWS_HELD_HANDLE_DENY_WRITE_DELETE' if os.name == 'nt'
                               else 'POSIX_HELD_FD_AUTHENTICATED_BLOCK_READS',
                'block_bytes': self.block_bytes, 'blocks': len(self.blocks),
                'posix_global_write_exclusion_claimed': False}

    def close(self):
        if not getattr(self,'closed', True):
            self.closed = True
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def __del__(self):
        self.close()
