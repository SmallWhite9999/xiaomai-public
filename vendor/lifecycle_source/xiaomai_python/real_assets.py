"""Read-only admission of the exact ORNITH ancestor snapshot. Standard library only.

Pins are derived from the fixed repository MODEL_RUNTIME_MANIFEST.json, not
from filenames supplied by a caller. This is NOT admission of a current True
checkpoint or proof that any model has run. Test manifests use verify_files(),
never the production verify_snapshot() entry.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import time
from .long_paths import asset_child, local_path, no_links

MODEL_ID = 'ornith-ai/Ornith-1.5-9B'
REVISION = '489cb97981b8654bcfcf30ce1f94ed1b62e07b53'
SOURCE_COMMIT = '9665f56519364b00448513952d918656c3381d3e'
SOURCE_PATH = 'artifacts/ornith_experience_learning_v1/prebuild/MODEL_RUNTIME_MANIFEST.json'
SOURCE_GIT_BLOB = '68ce0640338a56a0282e7e6a707bcf5ca6609a09'
METADATA = {
 'config.json': '1f1b3751c38f16a63340df90a55e870bef0f0b2968d833825a605b7cf930a313',
 'model.safetensors.index.json': 'd5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3',
 'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
 'chat_template.jinja': '9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235',
 'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
 'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
 'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}
SHARDS = {
 'model-00001-of-00004.safetensors': (4911674496, '7cf17ccbef5108b40d3b5f6418d4ef1a1c2f2a9c96791be893f4d510cfa966e4'),
 'model-00002-of-00004.safetensors': (4954211152, 'd78128c42a7de727b3becbe25d78b3374729abc0635bbb822f66f675107df3c2'),
 'model-00003-of-00004.safetensors': (4988356912, '5503caeb27904e56c5f8e4d62a37282da79ed41d6dc2da0ae57b135460c8cdf2'),
 'model-00004-of-00004.safetensors': (4452061304, 'd3faa4ac32ae58dca23480832953edf26f0016b65ac7e0702feeb59a6c73eb2b'),
}
PINNED_FILES = {**{n: {'sha256': h, 'bytes': None} for n,h in METADATA.items()},
                **{n: {'sha256': h, 'bytes': s} for n,(s,h) in SHARDS.items()}}
WIDTHS = {'BOOL':1, 'U8':1, 'I8':1, 'I16':2, 'U16':2, 'F16':2, 'BF16':2,
          'I32':4, 'U32':4, 'F32':4, 'I64':8, 'U64':8, 'F64':8}

from .source_handles import AssetError, PinnedReader, stat_identity


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def json_data(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda v: (_ for _ in ()).throw(AssetError('NONFINITE_JSON:' + v)))


def fingerprint(path):
    return stat_identity(Path(path).stat())


def checked_digest(path, *, expected_sha256=None, expected_bytes=None, tick=None):
    with PinnedReader(path, expected_sha256=expected_sha256, expected_bytes=expected_bytes, tick=tick) as reader:
        return reader.receipt()


def verify_files(root, specs, *, tick=None, progress=None):
    """Generic integrity primitive. Does not label caller-supplied pins as ORNITH."""
    root = no_links(root)
    if not root.is_dir():
        raise AssetError('SNAPSHOT_DIRECTORY_REQUIRED')
    if type(specs) is not dict or not specs:
        raise AssetError('NONEMPTY_PIN_SET_REQUIRED')
    found = {}
    for name, spec in specs.items():
        sha = spec['sha256']
        if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise AssetError('INVALID_SHA256_PIN')
        p = asset_child(root, name)
        found[name] = checked_digest(p, expected_sha256=sha, expected_bytes=spec.get('bytes'), tick=tick)
        if progress:
            progress({'event': 'FILE_VERIFIED', 'file': name, 'bytes': found[name]['bytes']})
    for name, info in found.items():
        if list(fingerprint(asset_child(root, name))) != info['stat']:
            raise AssetError('SOURCE_CHANGED_DURING_SET_VERIFICATION:' + name)
    return found


class TensorHeader:
    """Parse offsets without loading tensor payloads or executing serialized code."""
    def __init__(self, path, *, reader=None):
        self.path = no_links(path)
        self.reader = reader or PinnedReader(self.path)
        self.stat = self.reader.stat
        if True:
            prefix = self.reader.read(0, min(8,self.stat[2]))
            if len(prefix) != 8:
                raise AssetError('SHORT_SAFETENSORS')
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= 64 << 20 or 8 + length > self.stat[2]:
                raise AssetError('INVALID_SAFETENSORS_HEADER_LENGTH')
            raw = self.reader.read(8, length)
        self.header_sha256 = hashlib.sha256(raw).hexdigest()
        self.data_start = 8 + length
        data = json_data(raw)
        if type(data) is not dict:
            raise AssetError('INVALID_SAFETENSORS_HEADER')
        self.rows = {}
        for name, row in data.items():
            if name == '__metadata__':
                if type(row) is not dict or any(type(k) is not str or type(v) is not str for k,v in row.items()):
                    raise AssetError('INVALID_METADATA')
                continue
            if type(row) is not dict or set(row) != {'dtype','shape','data_offsets'}:
                raise AssetError('INVALID_TENSOR_HEADER_ROW')
            dt, shape, offsets = row['dtype'], row['shape'], row['data_offsets']
            if dt not in WIDTHS or type(shape) is not list or any(type(x) is not int or x < 0 for x in shape):
                raise AssetError('INVALID_TENSOR_DTYPE_SHAPE')
            if type(offsets) is not list or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
                raise AssetError('INVALID_TENSOR_OFFSETS')
            start, end = offsets
            count = math.prod(shape)
            if end - start != count * WIDTHS[dt] or end + self.data_start > self.stat[2]:
                raise AssetError('TENSOR_BYTE_COUNT_MISMATCH')
            self.rows[name] = {**row, 'elements':count, 'absolute_offset':self.data_start+start,
                               'tensor_bytes':end-start, 'shard':self.path.name}
        end = 0
        for row in sorted(self.rows.values(), key=lambda r: tuple(r['data_offsets'])):
            start, stop = row['data_offsets']
            if start != end:
                raise AssetError('DATA_HOLE_OR_OVERLAP')
            end = stop
        if end + self.data_start != self.stat[2]:
            raise AssetError('UNINDEXED_TENSOR_BYTES')
        self.recheck()

    def recheck(self):
        self.reader._assert_path()

    def read(self, name, *, start=0, count=None):
        self.recheck()
        row = self.rows[name]
        if count is None:
            count = row['elements'] - start
        if type(start) is not int or type(count) is not int or min(start,count) < 0 or start+count > row['elements']:
            raise AssetError('TENSOR_SLICE_OUT_OF_RANGE')
        width = WIDTHS[row['dtype']]
        raw = self.reader.read(row['absolute_offset'] + start*width, count*width)
        self.recheck()
        if len(raw) != count*width:
            raise AssetError('TRUNCATED_TENSOR')
        return raw


@dataclass
class VerifiedSnapshot:
    root: Path
    files: dict
    tensors: dict
    headers: dict
    config: dict
    receipt: dict
    readers: dict | None = None

    def close(self):
        for reader in (self.readers or {}).values(): reader.close()

    def __enter__(self): return self

    def __exit__(self, *args): self.close()

    def metadata(self, name):
        if not self.readers or name not in self.readers:
            raise AssetError("UNPINNED_METADATA_READER")
        return self.readers[name].read()

    def verify_all(self):
        return {name:r.verify_all() for name,r in self.readers.items()}

    def recheck(self):
        for name, info in self.files.items():
            if self.readers: self.readers[name]._assert_path()
            if list(fingerprint(asset_child(self.root, name))) != info['stat']:
                raise AssetError('SOURCE_CHANGED_SINCE_ADMISSION:' + name)

    def raw(self, name, *, start=0, count=None):
        self.recheck()
        return self.headers[self.tensors[name]['shard']].read(name, start=start, count=count)


def assemble_verified(root, files, shard_names, *, identity, readers=None):
    """Shared assembly for tests and production; identity is explicitly recorded."""
    root = no_links(root)
    if readers is None:
        readers = {}
        try:
            for name, info in files.items():
                readers[name] = PinnedReader(asset_child(root,name), expected_sha256=info['sha256'], expected_bytes=info['bytes'])
        except BaseException:
            for r in readers.values(): r.close()
            raise
    tensors = {}; headers = {}
    for name in shard_names:
        if name not in files:
            raise AssetError('UNVERIFIED_SHARD')
        header = TensorHeader(asset_child(root, name), reader=readers[name])
        if list(header.stat) != files[name]['stat']:
            raise AssetError('SOURCE_CHANGED_BEFORE_HEADER_READ')
        headers[name] = header
        for tname, row in header.rows.items():
            if tname in tensors:
                raise AssetError('DUPLICATE_TENSOR_ACROSS_SHARDS')
            tensors[tname] = row
    idx = json_data(readers['model.safetensors.index.json'].read())
    if idx.get('weight_map') != {n:r['shard'] for n,r in tensors.items()}:
        raise AssetError('INDEX_HEADER_MISMATCH')
    total = sum(r['tensor_bytes'] for r in tensors.values())
    if idx.get('metadata', {}).get('total_size') != total:
        raise AssetError('INDEX_TOTAL_BYTES_MISMATCH')
    cfg = json_data(readers['config.json'].read())
    receipt = {'status':'ANCESTOR_ASSETS_VERIFIED' if identity == REVISION else 'TEST_ASSETS_VERIFIED',
               'identity':identity, 'model_id':MODEL_ID if identity == REVISION else None,
               'files':files, 'tensor_count':len(tensors), 'tensor_bytes':total,
               'current_true_verified':False, 'model_executed':False,
               'scope':'DECLARED_ANCESTOR_ASSETS_NOT_WHOLE_DIRECTORY_OR_TRUE_CHECKPOINT'}
    result = VerifiedSnapshot(root, files, tensors, headers, cfg, receipt, readers)
    result.recheck()
    return result


def verify_snapshot(root, *, tick=None, progress=None):
    root = no_links(root)
    readers = {}
    try:
        for name, info in PINNED_FILES.items():
            readers[name] = PinnedReader(asset_child(root,name), expected_sha256=info['sha256'],
                expected_bytes=info['bytes'], tick=tick, progress=progress)
        files = {name:r.receipt() for name,r in readers.items()}
        result = assemble_verified(root, files, SHARDS, identity=REVISION, readers=readers)
    except BaseException:
        for r in readers.values(): r.close()
        raise
    cfg = result.config
    tc = cfg.get('text_config', {})
    if cfg.get('model_type') != 'qwen3_5' or tc.get('hidden_size') != 4096 or tc.get('num_hidden_layers') != 32:
        raise AssetError('PINNED_ARCHITECTURE_MISMATCH')
    if result.receipt['tensor_bytes'] != 19306208736:
        raise AssetError('PINNED_TENSOR_BYTES_MISMATCH')
    result.receipt.update(source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH, source_git_blob=SOURCE_GIT_BLOB)
    return result
