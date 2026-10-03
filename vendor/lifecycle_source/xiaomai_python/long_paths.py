"""Explicit Windows long paths and non-executing, symlink-safe file handling."""
from __future__ import annotations
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import stat


def extended_windows_path(value: str) -> str:
    """Pure path conversion; does not contact a UNC server or modify a file."""
    if value.startswith('\\\\?\\'):
        return value
    if not ntpath.isabs(value):
        raise ValueError('ABSOLUTE_WINDOWS_PATH_REQUIRED')
    value = ntpath.normpath(value)
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\' + value[2:]
    drive, tail = ntpath.splitdrive(value)
    if not drive or not tail.startswith('\\'):
        raise ValueError('ABSOLUTE_WINDOWS_PATH_REQUIRED')
    return '\\\\?\\' + value


def local_path(value) -> Path:
    value = os.path.abspath(os.fspath(value))
    return Path(extended_windows_path(value) if os.name == 'nt' else value)


def no_links(path: Path) -> Path:
    path = local_path(path)
    # Do not call resolve() first: it would hide a symlink/reparse point.
    for part in reversed([path, *path.parents]):
        s = part.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 0x400:
            raise ValueError('LINK_OR_REPARSE_POINT_REJECTED:' + str(part))
    return path


def asset_child(root, name: str) -> Path:
    if type(name) is not str or not name or '\\' in name or ':' in name:
        raise ValueError('UNSAFE_ASSET_NAME')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ValueError('UNSAFE_ASSET_NAME')
    return no_links(local_path(root).joinpath(*p.parts))


def mkdir_safe(path):
    """Check all existing ancestors before creating anything through them."""
    path = local_path(path)
    pending = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        pending.append(cursor)
        if cursor.parent == cursor: raise ValueError('NO_EXISTING_PARENT')
        cursor = cursor.parent
    no_links(cursor)
    if not cursor.is_dir(): raise ValueError('DIRECTORY_REQUIRED')
    for target in reversed(pending):
        no_links(target.parent)
        target.mkdir(exist_ok=False)
        no_links(target)
    return path


def write_json_new(path, value) -> Path:
    """Exclusive creation: refuses to overwrite earlier evidence."""
    path = local_path(path)
    mkdir_safe(path.parent)
    no_links(path.parent)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,'O_NOFOLLOW',0)
    with os.fdopen(os.open(path,flags,0o600),'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return path
