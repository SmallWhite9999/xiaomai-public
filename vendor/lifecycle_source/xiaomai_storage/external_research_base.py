"""Explicit immutable external-base binding for research-only checkpoints.

The fixed snapshot is reopened and verified by the caller in each process.
Construction independently compares every frozen model value against declared
snapshot segments. Subsequent checkpoints hash the whole live base in bounded
chunks. Only hashes/geometry/source references enter the readable checkpoint;
the frozen base is never silently omitted or reconstructed from this payload.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch
from . import kernels as k
from .research_checkpoint import ResearchCheckpointError, _configuration, _model_parts, _topology, _plain
from xiaomai_python.real_assets import VerifiedSnapshot, PINNED_FILES, REVISION, MODEL_ID

CHUNK_BYTES = 4 << 20
CHUNK_ELEMENTS = CHUNK_BYTES // 4


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _storage_digest(tensor, tick):
    raw = k.raw(tensor)
    digest = hashlib.sha256()
    for start in range(0, len(raw), CHUNK_BYTES):
        if tick is not None: tick()
        digest.update(raw[start:start + CHUNK_BYTES])
    return digest.hexdigest()


class FrozenExternalBase:
    """Process-local verifier; serialize only its verified plain-data manifest."""

    @classmethod
    def from_snapshot(cls, model, runtime, snapshot, *, sources, tick=None):
        """Production admission accepts only the independently pinned ancestor."""
        return cls(model, runtime, snapshot, sources=sources, tick=tick, _test=False)

    @classmethod
    def for_test_snapshot(cls, model, runtime, snapshot, *, sources, tick=None):
        """Explicit synthetic-snapshot tests, never labelled real asset custody."""
        return cls(model, runtime, snapshot, sources=sources, tick=tick, _test=True)

    def __init__(self, model, runtime, snapshot, *, sources, tick, _test):
        if type(snapshot) is not VerifiedSnapshot:
            raise ResearchCheckpointError('VERIFIED_SNAPSHOT_HANDLE_REQUIRED')
        if type(_test) is not bool:
            raise ResearchCheckpointError('EXPLICIT_EXTERNAL_BASE_MODE_REQUIRED')
        self._model, self._runtime, self._snapshot = model, runtime, snapshot
        self._test, self._tick = _test, tick
        self._sources = _plain(sources)
        self._source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._snapshot_identity = self._snapshot_manifest()
        # Rehash source files once at binding, rather than trusting a stale
        # caller receipt. The caller retains the verified handles until done.
        snapshot.verify_all()
        parameters, buffers = _model_parts(model)
        research_parameters = {'research_neural.' + name: p
            for name, p in runtime.components.named_parameters(remove_duplicate=False)}
        research_buffers = {'research_neural.' + name: p
            for name, p in runtime.components.named_buffers(remove_duplicate=False)}
        if any(parameters.get(n) is not p for n, p in research_parameters.items()) or any(
                buffers.get(n) is not p for n, p in research_buffers.items()):
            raise ResearchCheckpointError('RESEARCH_COMPONENT_BINDING_CHANGED')
        self.research_parameters = frozenset(research_parameters)
        self.research_buffers = frozenset(research_buffers)
        self._base = {n: p for n, p in parameters.items() if n not in self.research_parameters}
        if set(self._base) & set(buffers):
            raise ResearchCheckpointError('AMBIGUOUS_FROZEN_NAME')
        self._base.update((n, p) for n, p in buffers.items() if n not in self.research_buffers)
        if not self._base or set(self._sources) != set(self._base):
            raise ResearchCheckpointError('COMPLETE_FROZEN_SOURCE_MAPPING_REQUIRED')
        self._bindings = self._binding_ids(parameters, buffers)
        self._all_topology = _topology(parameters, buffers)
        self._configuration = _configuration(model, runtime)
        self._assembly = self._assembly_metadata()
        self._check_frozen(parameters, buffers)
        source_rows = {}
        for name, target in self._base.items():
            source_rows[name] = self._compare_source(name, target)
        self._hashes = self._live_hashes()
        self._manifest = {
            'schema': 'XIAOMAI_EXTERNAL_FROZEN_BASE_V1',
            'real_assets': not self._test,
            'snapshot': self._snapshot_identity,
            'sources': self._sources,
            'source_comparisons': source_rows,
            'full_model_topology': self._all_topology,
            'configuration_sha256': _digest(self._configuration),
            'assembly_metadata': self._assembly,
            'storage_sha256': self._hashes,
            'base_parameter_names': sorted(set(parameters) - self.research_parameters),
            'base_buffer_names': sorted(set(buffers) - self.research_buffers),
            'research_parameter_names': sorted(self.research_parameters),
            'research_buffer_names': sorted(self.research_buffers),
            'verifier_source_sha256': self._source_hash,
            'conversion': 'Pinned source dtype converted in bounded CPU torch chunks, same loading contract; not a Python arithmetic rewrite claim.',
            'restoration': 'Fresh process must verify snapshot, reconstruct the entire frozen base, then bind and compare this manifest.',
            'source_recheck': 'Full source hash at binding; persistent readers and consumed-block checks; caller must final-verify source custody.',
        }
        self.verify(model, runtime)

    @staticmethod
    def _binding_ids(parameters, buffers):
        return tuple((kind, name, id(t), t.untyped_storage()._cdata)
                     for kind, values in (('parameters', parameters), ('buffers', buffers))
                     for name, t in values.items())

    def _assembly_metadata(self):
        """Bind actual sparse-row lookup semantics, not an unused config copy."""
        from tools.real_research_fixture import DeclaredEmbeddingRows, RealLayerResearchModel
        from xiaomai_python.real_layer import EMBEDDING
        model = self._model
        backbone = model.model.language_model
        embedding = backbone.embed_tokens
        if type(embedding) is not DeclaredEmbeddingRows:
            if not self._test:
                raise ResearchCheckpointError('DECLARED_REAL_EMBEDDING_ASSEMBLY_REQUIRED')
            return {'kind': 'synthetic_complete_embedding'}
        if not self._test and type(model) is not RealLayerResearchModel:
            raise ResearchCheckpointError('EXACT_REAL_LAYER_RESEARCH_ASSEMBLY_REQUIRED')
        if backbone.config is not model.config:
            raise ResearchCheckpointError('BACKBONE_CONFIG_BINDING_CHANGED')
        ids, vocabulary, width = embedding.declared_ids, embedding.num_embeddings, embedding.embedding_dim
        if (type(ids) is not tuple or not ids or any(type(n) is not int for n in ids)
                or len(set(ids)) != len(ids) or type(vocabulary) is not int or type(width) is not int
                or any(n < 0 or n >= vocabulary for n in ids)):
            raise ResearchCheckpointError('DECLARED_EMBEDDING_METADATA_DOMAIN')
        row = self._snapshot.tensors.get(EMBEDDING)
        if row is None or row['shape'] != [vocabulary, width] or list(embedding.weight.shape) != [len(ids), width]:
            raise ResearchCheckpointError('DECLARED_EMBEDDING_SOURCE_SHAPE_MISMATCH')
        expected = {'segments': [{'tensor': EMBEDDING, 'start': n * width, 'count': width} for n in ids]}
        if self._sources.get('model.language_model.embed_tokens.weight') != expected:
            raise ResearchCheckpointError('DECLARED_EMBEDDING_ID_SOURCE_MAPPING_MISMATCH')
        return {'kind': 'declared_real_embedding_rows', 'declared_ids': list(ids),
                'num_embeddings': vocabulary, 'embedding_dim': width}

    def _snapshot_manifest(self):
        snapshot = self._snapshot
        snapshot.recheck()
        if not snapshot.readers or set(snapshot.readers) != set(snapshot.files):
            raise ResearchCheckpointError('LIVE_PINNED_SOURCE_HANDLES_REQUIRED')
        receipt = snapshot.receipt
        if not self._test:
            if receipt.get('identity') != REVISION or receipt.get('model_id') != MODEL_ID or receipt.get('status') != 'ANCESTOR_ASSETS_VERIFIED':
                raise ResearchCheckpointError('PINNED_ANCESTOR_IDENTITY_REQUIRED')
            if set(snapshot.files) != set(PINNED_FILES):
                raise ResearchCheckpointError('PINNED_SNAPSHOT_FILE_SET_MISMATCH')
            for name, pin in PINNED_FILES.items():
                row = snapshot.files[name]
                if row['sha256'] != pin['sha256'] or (pin['bytes'] is not None and row['bytes'] != pin['bytes']):
                    raise ResearchCheckpointError('PINNED_SNAPSHOT_FILE_MISMATCH:' + name)
        elif receipt.get('identity') == REVISION or receipt.get('status') != 'TEST_ASSETS_VERIFIED':
            raise ResearchCheckpointError('TEST_SNAPSHOT_MUST_NOT_CLAIM_REAL_ASSETS')
        config_bytes = snapshot.metadata('config.json')
        if _plain(json.loads(config_bytes)) != _plain(snapshot.config):
            raise ResearchCheckpointError('SNAPSHOT_CONFIG_CONTENT_CHANGED')
        return {'identity': receipt['identity'], 'model_id': receipt['model_id'],
                'status': receipt['status'],
                'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                'files': {n: {'sha256': r['sha256'], 'bytes': r['bytes']}
                          for n, r in sorted(snapshot.files.items())}}

    def _check_frozen(self, parameters, buffers, optimizer=None):
        if self._binding_ids(parameters, buffers) != self._bindings:
            raise ResearchCheckpointError('EXTERNAL_BASE_OR_RESEARCH_OBJECT_IDENTITY_CHANGED')
        if _topology(parameters, buffers) != self._all_topology:
            raise ResearchCheckpointError('EXTERNAL_BASE_TOPOLOGY_CHANGED')
        research_stores = {p.untyped_storage()._cdata for n, p in parameters.items() if n in self.research_parameters}
        research_stores.update(p.untyped_storage()._cdata for n, p in buffers.items() if n in self.research_buffers)
        for name, tensor in self._base.items():
            k.admit(tensor)
            if tensor.requires_grad or tensor.grad is not None:
                raise ResearchCheckpointError('EXTERNAL_BASE_MUST_REMAIN_FROZEN_WITHOUT_GRADIENT:' + name)
            if tensor.untyped_storage()._cdata in research_stores:
                raise ResearchCheckpointError('RESEARCH_BASE_STORAGE_ALIAS_NOT_ADMITTED')
            if (not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes()):
                raise ResearchCheckpointError('EXTERNAL_BASE_COMPLETE_DENSE_STORAGE_REQUIRED:' + name)
        if optimizer is not None and not set(optimizer.named).issubset(self.research_parameters):
            raise ResearchCheckpointError('EXTERNAL_BASE_OPTIMIZER_MUST_ONLY_OWN_RESEARCH')

    def _compare_source(self, name, target):
        spec = self._sources[name]
        if type(spec) is not dict or set(spec) != {'segments'} or type(spec['segments']) is not list or not spec['segments']:
            raise ResearchCheckpointError('EXTERNAL_SOURCE_SEGMENTS_REQUIRED:' + name)
        destination = k.raw(target)
        offset = 0
        records = []
        dtypes = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64}
        if not any(target.dtype is dt for dt in dtypes.values()):
            raise ResearchCheckpointError('EXTERNAL_LOADED_DTYPE_NOT_ADMITTED:' + name)
        for segment in spec['segments']:
            if type(segment) is not dict or set(segment) != {'tensor', 'start', 'count'}:
                raise ResearchCheckpointError('EXTERNAL_SOURCE_SEGMENT_SCHEMA')
            tensor_name, start, count = segment['tensor'], segment['start'], segment['count']
            if type(tensor_name) is not str or tensor_name not in self._snapshot.tensors or type(start) is not int or type(count) is not int or start < 0 or count < 1:
                raise ResearchCheckpointError('EXTERNAL_SOURCE_SEGMENT_RANGE')
            row = self._snapshot.tensors[tensor_name]
            if row['dtype'] not in dtypes or start + count > __import__('math').prod(row['shape']):
                raise ResearchCheckpointError('EXTERNAL_SOURCE_DTYPE_OR_RANGE')
            raw_hash, converted_hash = hashlib.sha256(), hashlib.sha256()
            for cursor in range(start, start + count, CHUNK_ELEMENTS):
                if self._tick is not None: self._tick()
                amount = min(CHUNK_ELEMENTS, start + count - cursor)
                raw = self._snapshot.raw(tensor_name, start=cursor, count=amount)
                raw_hash.update(raw)
                converted = torch.frombuffer(bytearray(raw), dtype=dtypes[row['dtype']]).to(target.dtype)
                bits = k.raw(converted)
                end = offset + len(bits)
                if end > len(destination) or destination[offset:end] != bits:
                    raise ResearchCheckpointError('EXTERNAL_BASE_VALUE_DOES_NOT_MATCH_SOURCE:' + name)
                converted_hash.update(bits)
                offset = end
            records.append({**segment, 'raw_sha256': raw_hash.hexdigest(),
                            'loaded_sha256': converted_hash.hexdigest(), 'loaded_dtype': str(target.dtype)})
        if offset != len(destination):
            raise ResearchCheckpointError('EXTERNAL_SOURCE_DOES_NOT_COVER_FULL_STORAGE:' + name)
        return records

    def _live_hashes(self):
        hashes, stores = {}, {}
        for name, tensor in self._base.items():
            sid = tensor.untyped_storage()._cdata
            if sid not in stores: stores[sid] = _storage_digest(tensor, self._tick)
            hashes[name] = stores[sid]
        return hashes

    def verify(self, model, runtime, optimizer=None):
        if model is not self._model or runtime is not self._runtime or runtime.model is not model:
            raise ResearchCheckpointError('EXTERNAL_BASE_RUNTIME_BINDING_CHANGED')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._source_hash:
            raise ResearchCheckpointError('EXTERNAL_BASE_VERIFIER_SOURCE_CHANGED')
        if self._snapshot_manifest() != self._snapshot_identity:
            raise ResearchCheckpointError('EXTERNAL_SNAPSHOT_IDENTITY_CHANGED')
        parameters, buffers = _model_parts(model)
        self._check_frozen(parameters, buffers, optimizer)
        if _configuration(model, runtime) != self._configuration:
            raise ResearchCheckpointError('EXTERNAL_BASE_CONFIGURATION_OR_SOURCE_CHANGED')
        if self._assembly_metadata() != self._assembly:
            raise ResearchCheckpointError('EXTERNAL_BASE_ASSEMBLY_METADATA_CHANGED')
        if self._live_hashes() != self._hashes:
            raise ResearchCheckpointError('EXTERNAL_FROZEN_BASE_CONTENT_CHANGED')
        # JSON roundtrip protects the verifier's stored manifest from caller edits.
        return json.loads(json.dumps(self._manifest, ensure_ascii=False, allow_nan=False))
