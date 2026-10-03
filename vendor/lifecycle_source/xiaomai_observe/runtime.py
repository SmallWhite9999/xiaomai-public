"""Scoped Python/module/ATen observation, with explicit native/GPU boundaries.

No module class-name filters, no modulo sampling, no swallowed recording errors.
This is an eager diagnostic implementation; torch.compile/cudagraph internals
are NOT Python-line events. Existing threads require explicit enrollment.
"""
from __future__ import annotations
from xiaomai_observe.code_identity import code_fingerprint
import contextlib
import contextvars
import dis
import functools
import inspect
import json
import marshal
import os
import struct
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Callable
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .passive import exception_info
from .journal import Journal,EvidenceError,new_file,canonical,digest
from .capture import BUSY,Capture,measurement,source_identity,model_inventory
from .source_binding import SourceAudit

_SCOPE=contextvars.ContextVar('xmo_scope',default=())
_ACTIVE_MODELS=set()
_ACTIVE_LOCK=threading.RLock()
_OPERATOR_MONITORS=contextvars.ContextVar('xmo_operator_monitors',default=())
_OPERATOR_CALLS=contextvars.ContextVar('xmo_operator_calls',default=())


def _same_operator_argument(left,right):
    if left is right:return True
    kind=type(left)
    if kind is not type(right):return False
    # The dispatcher re-materializes builtin shape lists when forwarding.
    # Compare only exact inert types; never invoke Tensor/custom __eq__.
    if any(kind is t for t in (str,int,bool,type(None))):return left==right
    if kind is float:return struct.pack('<d',left)==struct.pack('<d',right)
    if any(kind is t for t in (tuple,list)):
        return len(left)==len(right) and all(_same_operator_argument(a,b) for a,b in zip(left,right))
    return False


def _same_operator_call(frame,monitor,func,args,kwargs):
    # Tensor equality can execute code and is not an invocation identity check.
    old_monitor,old_func,old_args,old_kwargs=frame
    return (old_monitor is monitor and old_func is func and len(old_args)==len(args)
            and all(_same_operator_argument(a,b) for a,b in zip(old_args,args))
            and old_kwargs.keys()==kwargs.keys()
            and all(_same_operator_argument(old_kwargs[key],value) for key,value in kwargs.items()))


@contextlib.contextmanager
def operator_observation(func,args,kwargs,*,monitor=None):
    """Observe the actual ATen invocation even when a higher mode consumes it.

    PythonReferenceBackend calls this bridge before its own implementation. A
    lower OperatorMode sees the same invocation only when it is forwarded; that
    visit is deduplicated. Different nested invocations retain separate events.
    Capture itself stays under BUSY and never becomes an observed model call.
    """
    if BUSY.get():
        yield ()
        return
    kwargs=kwargs or {}
    monitors=(monitor,) if monitor is not None else _OPERATOR_MONITORS.get()
    previous=_OPERATOR_CALLS.get();frames=[];calls=[]
    with contextlib.ExitStack() as stack:
        for current in monitors:
            if any(_same_operator_call(frame,current,func,args,kwargs) for frame in previous+tuple(frames)):
                continue
            call=stack.enter_context(current.call('ATEN',str(func),{'args':args,'kwargs':kwargs},
                                                 source=current.current_source()))
            calls.append(call);frames.append((current,func,args,kwargs))
        token=_OPERATOR_CALLS.set(previous+tuple(frames))
        try:yield tuple(calls)
        finally:_OPERATOR_CALLS.reset(token)

def export_profiler_trace(profiler, destination):
    """Bridge native Kineto export to Python's Unicode-aware file writer.

    Some Windows Kineto builds cannot open a Unicode destination. Export once
    into a private temporary directory, validate the actual file, then publish
    its exact bytes. A missing/invalid native export remains a recording error.
    """
    with tempfile.TemporaryDirectory(prefix='xiaomai-profiler-') as staging:
        temporary = Path(staging) / 'trace.json'
        profiler.export_chrome_trace(str(temporary))
        raw = temporary.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get('traceEvents'), list):
            raise EvidenceError('INVALID_PROFILER_TRACE')
        new_file(Path(destination), raw)

def is_observing(model):
    with _ACTIVE_LOCK:return id(model) in _ACTIVE_MODELS

class OperatorMode(TorchDispatchMode):
    def __init__(self,monitor):super().__init__();self.monitor=monitor;self._tokens=[]
    def __enter__(self):
        result=super().__enter__()
        self._tokens.append(_OPERATOR_MONITORS.set(_OPERATOR_MONITORS.get()+(self.monitor,)))
        return result
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:_OPERATOR_MONITORS.reset(self._tokens.pop())
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if BUSY.get():return func(*args,**(kwargs or {}))
        with operator_observation(func,args,kwargs,monitor=self.monitor) as calls:
            label=('XMO_OP::'+calls[-1]['id']+'::'+str(func)) if calls else None
            with torch.profiler.record_function(label) if label else contextlib.nullcontext():
                output=func(*args,**(kwargs or {}))
            for call in calls:
                call['output']=output
                # Do not claim a CUDA kernel completed just because Python returned.
                call['completion']='CPU_API_RETURN_GPU_NOT_SYNCHRONIZED'
            return output

class LineTracer:
    def __init__(self,monitor,roots,*,capture_locals=True):
        self.m=monitor;self.roots=tuple(Path(p).resolve() for p in roots)
        self.capture_locals=capture_locals;self.frames={};self.observed=set();self.files={}
        self.self_root=Path(__file__).parent.resolve()
    def included(self,filename):
        if filename.startswith('<'):return False
        p=Path(filename).resolve()
        return not p.is_relative_to(self.self_root) and any(p.is_relative_to(r) if r.is_dir() else p==r for r in self.roots)
    def __call__(self,frame,event,arg):
        if BUSY.get():return self
        if not self.included(frame.f_code.co_filename):return None
        with measurement():
            key=id(frame);f=frame.f_code
            if key not in self.frames:
                self.frames[key]={'id':self.m.new_id('python'),'exception_seen':False}
            state=self.frames[key]
            location={'file':f.co_filename,'line':frame.f_lineno,'qualname':f.co_qualname}
            if f.co_filename not in self.files:
                try:
                    b=Path(f.co_filename).read_bytes()
                    ref=self.m.journal.blob(b)
                    self.files[f.co_filename]=ref['sha256']
                    self.m.journal.emit('PYTHON_SOURCE',file=f.co_filename,content=ref)
                except OSError as exc:
                    self.m.capture.gap('SOURCE_READ_FAILED',file=f.co_filename,error=type(exc).__name__)
            parent_frame=frame.f_back
            while parent_frame is not None and id(parent_frame) not in self.frames:
                parent_frame=parent_frame.f_back
            parent_id=self.frames[id(parent_frame)]['id'] if parent_frame is not None else None
            self.observed.add((f.co_filename,frame.f_lineno))
            payload={'call_id':state['id'],'parent_call_id':parent_id,
                     'parent_scope':list(_SCOPE.get()),**location,
                     'source_sha256':self.files.get(f.co_filename),
                     'loaded_code_object_sha256':code_fingerprint(f)}
            if event=='exception':
                state['exception_seen']=True
                info=exception_info(arg[1])
                payload.update(exception_type=info['type'],exception_message=info['message'],passive_exception=info)
            if self.capture_locals and event in ('call','line','return','exception'):
                payload['locals']=self.m.capture.tree(dict(frame.f_locals),'locals')
            if event=='return':
                # CPython emits return(None) when unwinding too; preserve ambiguity.
                op=f.co_code[frame.f_lasti] if 0<=frame.f_lasti<len(f.co_code) else None
                opname=dis.opname[op] if op is not None else None
                payload['opcode_at_event']=opname
                payload['return_or_yield']=self.m.capture.tree(arg,'return')
                payload['exception_seen_in_frame']=state['exception_seen']
                if opname not in ('YIELD_VALUE','YIELD_FROM'):self.frames.pop(key,None)
            self.m.journal.emit('PYTHON_'+event.upper(),**payload)
        return self
    @contextlib.contextmanager
    def activate(self):
        previous=sys.gettrace()
        if previous is not None:raise EvidenceError('EXISTING_TRACE_FUNCTION_NOT_OVERWRITTEN')
        sys.settrace(self)
        try:yield
        finally:
            replaced=sys.gettrace() is not self
            sys.settrace(previous)
            if replaced:
                self.m.capture.gap('PYTHON_TRACE_REPLACED_DURING_RUN')

class Monitor:
    """Observe a caller-owned model; never load/reset another True.

    full values captures every supported tensor value at selected observation
    boundaries. It is NOT all registers/temporaries inside fused CUDA kernels.
    Opaque Python values are recorded as gaps. A complete event chain is distinct
    from complete value/kernel coverage.
    """
    def __init__(self,model,root,*,run_id,python_roots=(),values='full',numerics=True,
                 operators=True,python_lines=True,capture_locals=True,profiler=False,
                 fsync_each=True, request_id=None, phase="diagnostic", producer_id=None,
                 python_instructions=False, require_python_scope=False, compact_locals=False,
                 value_transitions=False, native_calls=False):
        if (compact_locals or value_transitions or native_calls) and not python_instructions:
            raise EvidenceError('DETAILED_BOUNDARIES_REQUIRE_INSTRUCTION_MONITORING')
        if (compact_locals or value_transitions) and not capture_locals:
            raise EvidenceError('VALUE_BOUNDARIES_REQUIRE_CAPTURE_LOCALS')
        self.model=model;self.root=Path(root);self.run_id=run_id
        self.request_id=request_id;self.phase=phase;self.producer_id=producer_id
        self.options=dict(values=values,numerics=numerics,operators=operators,
                          python_lines=python_lines,profiler=profiler,
                          python_roots=[str(Path(r).resolve()) for r in python_roots],
                          capture_locals=capture_locals,python_instructions=python_instructions,
                          require_python_scope=require_python_scope,compact_locals=compact_locals,
                          value_transitions=value_transitions,native_calls=native_calls)
        self.fsync_each=fsync_each;self._counter=0;self._lock=threading.Lock()
        self.calls={};self.module_calls={};self._originals=[];self.thread_ids=set();self.unenrolled_threads=set();self._entered=False
    def new_id(self,prefix):
        with self._lock:
            n=self._counter;self._counter+=1
        return f'{prefix}:{n}'
    def current_source(self):
        f=sys._getframe(1)
        while f:
            filename=f.f_code.co_filename
            if not filename.startswith(str(Path(__file__).parent)) and '/torch/' not in filename:
                return {'file':filename,'line':f.f_lineno,'qualname':f.f_code.co_qualname}
            f=f.f_back
        return {'status':'SOURCE_NOT_FOUND'}
    @contextlib.contextmanager
    def call(self,kind,name,inputs,*,source=None):
        with measurement():
            cid=self.new_id(kind.lower());parents=_SCOPE.get()
            value={'id':cid};self.calls[cid]={'kind':kind,'name':name}
            self.journal.emit(kind+'_ENTER',call_id=cid,parent_call_id=parents[-1] if parents else None,
                name=name,source=source,inputs=self.capture.tree(inputs,'inputs'))
        token=_SCOPE.set(parents+(cid,))
        start=time.monotonic_ns()
        try:
            yield value
        except BaseException as exc:
            with measurement():
                try:
                    info=exception_info(exc)
                    self.journal.emit(kind+'_ERROR',call_id=cid,name=name,error_type=info['type'],
                        error_message=info['message'],passive_exception=info,cpu_elapsed_ns=time.monotonic_ns()-start)
                except EvidenceError as record_exc:
                    self.recording_failure=exception_info(record_exc) # do not mutate the observed exception
            raise
        else:
            with measurement():
                self.journal.emit(kind+'_RETURN',call_id=cid,name=name,
                    output=self.capture.tree(value.get('output'),'output'),
                    completion=value.get('completion','CPU_FUNCTION_RETURN'),
                    cpu_elapsed_ns=time.monotonic_ns()-start)
        finally:
            _SCOPE.reset(token);self.calls.pop(cid,None)
    def _wrap(self,name,module):
        original=module.forward
        had='forward' in module.__dict__;old=module.__dict__.get('forward')
        identity=source_identity(original)
        @functools.wraps(original)
        def observed(*args,**kwargs):
            if BUSY.get():return original(*args,**kwargs)
            tid=threading.get_ident()
            if tid not in self.thread_ids and tid not in self.unenrolled_threads:
                self.unenrolled_threads.add(tid)
                self.capture.gap('THREAD_NOT_ENROLLED_FOR_LINE_AND_OPERATOR_TRACE',thread_id=tid)
            self.module_calls[name]=self.module_calls.get(name,0)+1
            with self.call('MODULE',name,{'args':args,'kwargs':kwargs},source=identity) as call:
                result=original(*args,**kwargs)
                call['output']=result
                return result
        module.forward=observed
        self._originals.append((module,had,old,observed,name))
    def __enter__(self):
        if self._entered:raise EvidenceError('MONITOR_NOT_REENTRANT')
        with _ACTIVE_LOCK:
            if id(self.model) in _ACTIVE_MODELS:raise EvidenceError('MODEL_ALREADY_OBSERVED')
            _ACTIVE_MODELS.add(id(self.model))
        self._entered=True
        try:
            self.journal=Journal(self.root,run_id=self.run_id,fsync_each=self.fsync_each,
                                 request_id=self.request_id,phase=self.phase,producer_id=self.producer_id)
            self.capture=Capture(self.journal,values=self.options['values'],numerics=self.options['numerics'])
            if self.options['require_python_scope'] and self.options['python_lines'] and not self.options['python_roots']:
                raise EvidenceError('EMPTY_PYTHON_TRACE_SCOPE')
            if self.options['python_instructions']:
                if not self.options['python_lines']:raise EvidenceError('INSTRUCTION_REQUIRES_PYTHON_TRACE')
                from .instructions import InstructionTracer
                self.tracer=InstructionTracer(self,self.options['python_roots'],capture_locals=self.options['capture_locals'],require_roots=self.options['require_python_scope'])
            else:
                self.tracer=LineTracer(self,self.options['python_roots'],capture_locals=self.options['capture_locals'])
            with measurement():
                self.inventory=model_inventory(self.model)
                self.source_audit=SourceAudit()
                self.source_bindings=self.source_audit.model(self.model)
                self.journal.emit('SOURCE_BINDING_INVENTORY',binding=self.source_bindings)
                for row in self.source_bindings['modules']:
                    if row['binding']['status']!='EXECUTABLE_MATCH':
                        self.capture.gap('FORWARD_SOURCE_BINDING_NOT_MATCHED',module=row['name'],
                                         binding_status=row['binding']['status'])
                self.journal.emit('MODEL_INVENTORY',inventory=self.inventory,options=self.options)
                for name,module in self.model.named_modules():self._wrap(name,module)
            self.stack=contextlib.ExitStack()
            self.thread_ids.add(threading.get_ident())
            if self.options['profiler']:
                acts=[torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():acts.append(torch.profiler.ProfilerActivity.CUDA)
                self.prof=self.stack.enter_context(torch.profiler.profile(activities=acts,
                    record_shapes=True,with_stack=True))  # no one-step-only schedule
            if self.options['operators']:self.stack.enter_context(OperatorMode(self))
            if self.options['python_lines']:self.stack.enter_context(self.tracer.activate())
            return self
        except BaseException:
            if hasattr(self,'stack'):self.stack.close()
            self._restore()
            if hasattr(self,'journal'):self.journal.abort()
            with _ACTIVE_LOCK:_ACTIVE_MODELS.discard(id(self.model))
            raise
    def _restore(self):
        for module,had,old,wrapper,name in reversed(self._originals):
            if module.__dict__.get('forward') is not wrapper:
                if hasattr(self,'capture'):self.capture.gap('FORWARD_CHANGED_DURING_OBSERVATION',module=name)
                continue
            if had:module.forward=old
            else:del module.__dict__['forward']
        self._originals=[]
    @contextlib.contextmanager
    def thread_scope(self):
        """Explicitly enroll a worker thread; join it before leaving Monitor."""
        self.thread_ids.add(threading.get_ident())
        with contextlib.ExitStack() as stack:
            if self.options['operators']:stack.enter_context(OperatorMode(self))
            if self.options['python_lines']:stack.enter_context(self.tracer.activate())
            yield
    def checkpoint_event(self,label,value):
        with measurement():self.journal.emit('STATE_SNAPSHOT',label=label,value=self.capture.tree(value,label))
    def gradients(self):
        self.checkpoint_event('gradients',{n:p.grad for n,p in self.model.named_parameters()})
    def __exit__(self,exc_type,exc,tb):
        try:
            self.stack.__exit__(exc_type,exc,tb)
            with measurement():
                self._restore()
                if self.options['profiler']:
                    p=self.root/'operators.chrome.json'
                    export_profiler_trace(self.prof,p)
                    self.journal.emit('PROFILER_ARTIFACT',artifact=self.journal.artifact(p.name),
                                      boundary='SUPPORTED_KINETO_ACTIVITIES_NOT_ALL_HARDWARE_INSTRUCTIONS')
                for name,module in self.model.named_modules():
                    self.journal.emit('MODULE_COVERAGE',name=name,calls=self.module_calls.get(name,0),
                        state='EXECUTED' if self.module_calls.get(name,0) else 'NOT_EXECUTED')
                source_recheck=self.source_audit.recheck_files()
                self.journal.emit('SOURCE_FILES_RECHECK',files=source_recheck)
                for row in source_recheck:
                    if row['status']!='UNCHANGED':
                        self.capture.gap('SOURCE_FILE_CHANGED_OR_UNREADABLE',**row)
                coverage={'scope':'EXPLICIT_EAGER_RUN', 'module_wrappers':len(self.module_calls),
                    'module_call_counts':self.module_calls, 'source_binding_inventory':self.source_bindings,
                    'source_file_recheck':source_recheck, 'python_observed_line_locations':len(self.tracer.observed),
                    'python_source_roots':self.options['python_roots'],
                    'enrolled_threads':sorted(self.thread_ids),
                    'python_thread_policy':'EXPLICIT_ENROLLMENT_NOT_GLOBAL_THREAD_COVERAGE',
                    'tensor_values_mode':self.options['values'],'gaps':self.capture.gaps,
                    'unclosed_calls':list(self.calls),
                    'suspended_or_unfinished_python_frames':len(self.tracer.frames),
                    'gpu_kernel_full_values':False,
                    'compiled_python_line_coverage':False,'whole_true_verified':False,
                    'instructions':self.tracer.coverage() if hasattr(self.tracer,'coverage') else {'enabled':False}}
                self.result=self.journal.seal(execution='ERROR' if exc else 'RETURNED',coverage=coverage)
        except BaseException:
            self.journal.abort()
            if exc is not None:
                self.finalization_failed=True # original exception remains untouched; no success seal
                return False
            raise
        finally:
            self._restore()
            with _ACTIVE_LOCK:_ACTIVE_MODELS.discard(id(self.model))
        return False
