"""Explicit resource limits: limits yield a blocked run, never a partial PASS."""
from __future__ import annotations
from dataclasses import dataclass
import math
import shutil
import os
import stat
import time
from pathlib import Path

class ResourceLimit(RuntimeError):
    pass

@dataclass(frozen=True)
class Limits:
    seconds: float = 900.0
    max_rss_bytes: int = 8 << 30
    max_output_bytes: int = 2 << 30
    minimum_free_bytes: int = 1 << 30
    max_tokens: int = 2

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('POSITIVE_FINITE_SECONDS_REQUIRED')
        for name in ('max_rss_bytes','max_output_bytes','minimum_free_bytes','max_tokens'):
            v = getattr(self,name)
            if type(v) is not int or v <= 0:
                raise ValueError('POSITIVE_LIMIT_REQUIRED:' + name)


def directory_bytes(root):
    from .long_paths import no_links
    root = Path(root)
    if not root.exists(): return 0
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        # Recheck each directory before entering it, even if its name was just
        # enumerated. Never retain DirEntry metadata across budget checks.
        no_links(directory)
        current = directory.lstat()
        if stat.S_ISLNK(current.st_mode) or getattr(current,'st_file_attributes',0)&0x400:
            raise ResourceLimit('OUTPUT_LINK_OR_REPARSE_REJECTED')
        if not stat.S_ISDIR(current.st_mode):
            raise ResourceLimit('OUTPUT_DIRECTORY_REQUIRED')
        with os.scandir(directory) as entries:
            for entry in entries:
                s = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400:
                    raise ResourceLimit('OUTPUT_LINK_OR_REPARSE_REJECTED')
                if stat.S_ISREG(s.st_mode): total += s.st_size
                elif stat.S_ISDIR(s.st_mode): pending.append(Path(entry.path))
                else: raise ResourceLimit('OUTPUT_SPECIAL_FILE_REJECTED')
    return total

class Budget:
    def __init__(self, limits, output):
        self.limits, self.output, self.started = limits, Path(output), time.monotonic()
        self.peak_rss = 0
        self._last_disk_check = -float('inf')

    def _check_wall_and_rss(self):
        now = time.monotonic()
        if now-self.started > self.limits.seconds:
            raise ResourceLimit('WALL_TIME_LIMIT')
        import psutil
        rss = psutil.Process().memory_info().rss
        self.peak_rss = max(self.peak_rss,rss)
        if rss > self.limits.max_rss_bytes:
            raise ResourceLimit('RSS_LIMIT')
        now = time.monotonic()
        if now-self.started > self.limits.seconds:
            raise ResourceLimit('WALL_TIME_LIMIT')
        return now

    def check(self, *, force=False):
        # Time and resident memory are checked even between filesystem scans.
        now = self._check_wall_and_rss()
        if force or now-self._last_disk_check >= 0.25:
            if directory_bytes(self.output) > self.limits.max_output_bytes:
                raise ResourceLimit('OUTPUT_BYTES_LIMIT')
            parent = self.output
            while not parent.exists(): parent=parent.parent
            if shutil.disk_usage(parent).free < self.limits.minimum_free_bytes:
                raise ResourceLimit('FREE_DISK_LIMIT')
            # A slow scan consumes the run budget. Start the next interval only
            # after the scan succeeds and its elapsed time/RSS are rechecked.
            self._last_disk_check = self._check_wall_and_rss()
