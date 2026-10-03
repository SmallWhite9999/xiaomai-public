"""Reversible PythonTape gradient-control adapter for pinned research classes.

The original set_evidence/use_snapshot/activate/ResearchDecoderBlock, encoder,
state-core, bridge and scoring computations remain their pinned implementations.
Only the runtime's gradient-mode orchestration is replaced. Frozen feature
extraction is explicit; lower Python numerical dispatch remains enabled.

Use within one owned CPU thread:
    with PythonReferenceBackend(), PythonTape() as tape:
        with PythonResearchAdapter(runtime, tape=tape):
            with runtime.activate(training=True):
                loss = ...
    tape.backward(loss)
No native backward/grad is invoked, and the global grad guard is not relaxed.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import threading
import types
import torch
from .code_identity import code_fingerprint
from .python_autodiff import PythonTape, TapeError

SOURCE_BLOB='cb9c7574ccd22d8d7742c8c57de38178e7cbe5a9'
METHODS=('_state','score_candidates','_encode_scope','encode_only','use_snapshot',
         'activate','set_evidence','input_snapshot','_parameter_signature','_validate_snapshot')
REPLACED=('_state','score_candidates','_encode_scope')

class ResearchAdapterError(RuntimeError):pass

def _underlying(fn):
    if type(fn) is not types.FunctionType:raise ResearchAdapterError('EXACT_RESEARCH_METHOD_REQUIRED')
    seen=set()
    while '__wrapped__' in vars(fn):
        if id(fn) in seen:raise ResearchAdapterError('METHOD_WRAPPER_CYCLE')
        seen.add(id(fn));fn=vars(fn)['__wrapped__']
        if type(fn) is not types.FunctionType:raise ResearchAdapterError('EXACT_RESEARCH_METHOD_REQUIRED')
    return fn

def _expected_methods(path):
    raw=path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=SOURCE_BLOB:
        raise ResearchAdapterError('UNPINNED_RESEARCH_RUNTIME_SOURCE')
    module=compile(raw,str(path),'exec',dont_inherit=True)
    classes=[c for c in module.co_consts if type(c) is types.CodeType and c.co_name=='ResearchNeuralRuntime']
    if len(classes)!=1:raise ResearchAdapterError('RESEARCH_RUNTIME_CLASS_MISSING')
    return {c.co_name:code_fingerprint(c) for c in classes[0].co_consts if type(c) is types.CodeType}

def _state_entry(runtime,snapshot,*,training):
    return vars(runtime)['_python_research_adapter'].state(snapshot,training=training)

def _score_entry(runtime,token_rows,*,training=False):
    return vars(runtime)['_python_research_adapter'].score_candidates(token_rows,training=training)

def _encode_scope_entry(runtime):
    return vars(runtime)['_python_research_adapter'].encode_scope()

class PythonResearchAdapter:
    """One explicitly owned runtime and one active PythonTape; no global patch.

    Checkpoint only after leaving this context. Installation/removal invalidate
    disposable inference state, preserving input frontier and weight revision.
    Active leases/contexts or existing instance-method overrides are rejected.
    training=False freezes research state/scoring, as the original runtime does.
    The original snapshot context does not freeze the caller's decoder body.
    """
    def __init__(self,runtime,*,tape):
        if type(tape) is not PythonTape:raise ResearchAdapterError('EXACT_PYTHON_TAPE_REQUIRED')
        self.runtime,self.tape=runtime,tape
        self.source=Path(inspect.getfile(type(runtime))).resolve()
        self.expected=_expected_methods(self.source)
        self.original={name:getattr(type(runtime),name) for name in METHODS}
        for name,fn in self.original.items():
            if code_fingerprint(_underlying(fn).__code__)!=self.expected[name]:
                raise ResearchAdapterError('LOADED_RESEARCH_METHOD_MISMATCH:'+name)
        self.attached=False;self.owner_thread=None
        self.entries={'_state':_state_entry,'score_candidates':_score_entry,'_encode_scope':_encode_scope_entry}

    def _tape_ready(self):
        t=self.tape
        if not t._recording or t.closed or t.failed or threading.get_ident()!=t.owner_thread:
            raise ResearchAdapterError('ACTIVE_OWNER_PYTHON_TAPE_REQUIRED')
        if torch.is_grad_enabled():raise ResearchAdapterError('RESEARCH_ADAPTER_REQUIRES_NATIVE_GRAD_DISABLED')

    def _idle(self):
        r=self.runtime
        if r.binding.stack or r.binding.encode_depth:
            raise ResearchAdapterError('ACTIVE_RESEARCH_CONTEXT_NOT_ADMITTED')
        if getattr(r.model,'_ornith_research_cache_lease',None) is not None:
            raise ResearchAdapterError('ACTIVE_RESEARCH_CACHE_LEASE_NOT_ADMITTED')

    def _invalidate(self):
        self.runtime._inference_state=None;self.runtime._inference_key=None

    def check_binding(self):
        self._tape_ready()
        if not self.attached or vars(self.runtime).get('_python_research_adapter') is not self:
            raise ResearchAdapterError('RESEARCH_ADAPTER_BINDING_CHANGED')
        if threading.get_ident()!=self.owner_thread:raise ResearchAdapterError('RESEARCH_ADAPTER_THREAD_CHANGED')
        if _expected_methods(self.source)!=self.expected:raise ResearchAdapterError('RESEARCH_SOURCE_CHANGED')
        for name,fn in self.original.items():
            if getattr(type(self.runtime),name) is not fn:
                raise ResearchAdapterError('RESEARCH_CLASS_METHOD_CHANGED:'+name)
            if code_fingerprint(_underlying(fn).__code__)!=self.expected[name]:
                raise ResearchAdapterError('LOADED_RESEARCH_METHOD_CHANGED:'+name)
            if name in self.entries:
                actual=vars(self.runtime).get(name)
                if type(actual) is not types.MethodType or actual.__self__ is not self.runtime or actual.__func__ is not self.entries[name]:
                    raise ResearchAdapterError('RESEARCH_ADAPTER_METHOD_CHANGED:'+name)
            elif name in vars(self.runtime):
                raise ResearchAdapterError('RESEARCH_INSTANCE_METHOD_CHANGED:'+name)

    def __enter__(self):
        self._tape_ready();self._idle()
        if self.attached or '_python_research_adapter' in vars(self.runtime):
            raise ResearchAdapterError('RESEARCH_ADAPTER_ALREADY_ATTACHED')
        if any(name in vars(self.runtime) for name in METHODS):
            raise ResearchAdapterError('RESEARCH_INSTANCE_METHOD_OVERRIDE_NOT_ADMITTED')
        self.owner_thread=threading.get_ident()
        self._invalidate()
        self.runtime._python_research_adapter=self
        for name,fn in self.entries.items():setattr(self.runtime,name,types.MethodType(fn,self.runtime))
        self.attached=True
        try:
            self.check_binding()
            self.tape.emit('PY_RESEARCH_ADAPTER_ATTACHED',source_blob=SOURCE_BLOB,
                           native_backward=False,current_true_restored=False)
        except BaseException:
            self._remove();raise
        return self

    def _remove(self):
        # Remove only our own methods; never silently overwrite a concurrent edit.
        drift=[]
        for name,fn in self.entries.items():
            actual=vars(self.runtime).get(name)
            if type(actual) is types.MethodType and actual.__func__ is fn and actual.__self__ is self.runtime:
                delattr(self.runtime,name)
            else:drift.append(name)
        if vars(self.runtime).get('_python_research_adapter') is self:
            delattr(self.runtime,'_python_research_adapter')
        else:drift.append('_python_research_adapter')
        self.attached=False;self._invalidate()
        return drift

    def __exit__(self,exc_type,exc,tb):
        idle_error=None
        try:self._idle()
        except BaseException as error:idle_error=error
        drift=self._remove()
        if exc_type is None:
            if idle_error is not None:raise idle_error
            if drift:raise ResearchAdapterError('RESEARCH_METHOD_CHANGED_BEFORE_DETACH:'+','.join(drift))
            self.tape.emit('PY_RESEARCH_ADAPTER_DETACHED',inference_cache_invalidated=True)
        return False

    @contextmanager
    def encode_scope(self):
        self.check_binding()
        # This is the scope already placed by original encode_only around
        # frozen backbone features. Original eval/mode restoration is retained.
        with self.original['_encode_scope'](self.runtime):
            with self.tape.frozen_region(reason='PINNED_RESEARCH_FROZEN_BACKBONE_FEATURES'):
                yield

    def state(self,snapshot,*,training):
        self.check_binding()
        if type(training) is not bool:raise ResearchAdapterError('EXPLICIT_BOOL_TRAINING_REQUIRED')
        r=self.runtime
        # Same pinned _state orchestration; only native grad-mode switching is
        # replaced by explicit PythonTape training/frozen scopes.
        # inference_mode(False) itself restores grad-enabled mode. Keep normal
        # tensors while explicitly leaving derivative ownership with the Tape.
        with torch.inference_mode(False), torch.no_grad():
            signature=r._parameter_signature()
            key=(json.dumps(snapshot,sort_keys=True,separators=(',',':')),signature)
            if not training:
                with self.tape.frozen_region(reason='PINNED_RESEARCH_INFERENCE_STATE'):
                    if r._inference_key==key and r._inference_state is not None:
                        return r._inference_state.clone()
                    state=r.components.state_from_features(r.encode_only(snapshot['token_rows']))
                    r.state_recomputations+=1
                    state=state.detach()
                    r._inference_key,r._inference_state=key,state.clone()
                    return state
            state=r.components.state_from_features(r.encode_only(snapshot['token_rows']))
            r.state_recomputations+=1
            return state

    def score_candidates(self,token_rows,*,training=False):
        self.check_binding()
        if type(training) is not bool:raise ResearchAdapterError('EXPLICIT_BOOL_TRAINING_REQUIRED')
        r=self.runtime
        if not token_rows:raise ValueError('NO_USABLE_RESEARCH_CANDIDATES')
        if r.binding.stack:
            entry=r.binding.stack[-1]
            if entry['training']!=training:raise ValueError('candidate scoring must match active gradient context')
            if entry['signature']!=r._parameter_signature() or entry['state_version']!=entry['state']._version:
                raise RuntimeError('RESEARCH_STATE_STALE_DURING_ACTIVE_DECODE')
            state=entry['state']
        else:state=r._state(r.input_snapshot(),training=training)
        if training:return r.components.score(state,r.encode_only(token_rows))
        with self.tape.frozen_region(reason='PINNED_RESEARCH_INFERENCE_SCORING'):
            return r.components.score(state,r.encode_only(token_rows))
