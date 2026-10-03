"""In-memory copy of the admitted readable-data graph, not arbitrary deepcopy.

Python copies the storage bytes and graph; binding Tensor objects uses the same
explicit native boundary as literal_state.load. No Tensor.__deepcopy__, storage
clone kernel, pickle, save or load. Immutable cycles/custom objects are rejected.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from . import kernels as k
from . import literal_state as ls


def graph_signature(value, limits):
    graph,stores,owners=ls._build_graph(value,limits)
    encoded=ls._literal_bytes('STATE',graph)
    if len(encoded)>limits.max_graph_bytes:raise ls.StateError('GRAPH_SIZE_LIMIT')
    versions=[s['source']._version for s in stores]
    storage_hashes=[ls._source_hash(s,limits.block_bytes) for s in stores]
    ident=(ls.digest(encoded),tuple(id(o) for o in owners),
           tuple(s['source_storage']._cdata for s in stores),tuple(versions),tuple(storage_hashes))
    return graph,stores,owners,ident


def clone_data(value,*,limits=ls.Limits(),sink=None,after_copy=None):
    """Clone plain data without overwriting the input or calling custom methods.

    Concurrent writer exclusion is NOT provided. Observed source graph/bytes are
    checked before and after copying/binding. Failure yields no successful return.
    Version-counter values are not restored or compared across cloned objects.
    """
    source_paths=(Path(__file__),Path(ls.__file__),Path(k.__file__))
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    graph,stores,owners,before=graph_signature(value,limits)
    buffers=[];storage_objects=[];binding_owners=[]
    ls.emit(sink,'DATA_SNAPSHOT_BEGIN',nodes=len(graph['nodes']),storages=len(stores),
            bytes=sum(s['nbytes'] for s in stores),native_binding=True,
            version_counter_topology_restored=False)
    try:
        for sid,store in enumerate(stores):
            source=k.raw(store['source']);dest=bytearray(store['nbytes'])
            for at in range(0,len(source),limits.block_bytes):
                block=bytes(source[at:at+limits.block_bytes])
                ls.emit(sink,'DATA_SNAPSHOT_BLOCK',storage=sid,offset=at,source_hex=block.hex(),
                        bytes=len(block),sha256=ls.digest(block))
                dest[at:at+len(block)]=block
            if ls.digest(dest)!=before[-1][sid]:raise ls.StateError('SOURCE_CHANGED_DURING_SNAPSHOT')
            buffers.append(dest)
            with torch._C._DisableTorchDispatch():
                owner=torch.frombuffer(dest,dtype=torch.uint8) if dest else torch.empty(0,dtype=torch.uint8)
            binding_owners.append(owner);storage_objects.append(owner.untyped_storage())
        result=ls._decode_graph(graph,storage_objects,limits=limits,sink=sink)
        if after_copy is not None:after_copy()
        _,again,again_owners,after=graph_signature(value,limits)
        if before!=after:raise ls.StateError('SOURCE_CHANGED_BEFORE_SNAPSHOT_RETURN')
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=pins[str(p)] for p in source_paths):
            raise ls.StateError('SNAPSHOT_IMPLEMENTATION_CHANGED')
        ls.emit(sink,'DATA_SNAPSHOT_PREPARED',graph_sha256=before[0],source_hashes=list(before[-1]))
        # Observers are part of the experiment; detect changes they cause too.
        _,_,_,last=graph_signature(value,limits)
        if last!=before:raise ls.StateError('SOURCE_CHANGED_DURING_SNAPSHOT_OBSERVER')
        return result
    except BaseException:
        try:ls.emit(sink,'DATA_SNAPSHOT_FAILED',returned_success=False)
        except BaseException:pass
        raise
