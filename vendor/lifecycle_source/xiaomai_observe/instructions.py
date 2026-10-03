"""Per-instruction observation integrated with the existing Journal and Monitor.

No AST rewrite, eval, extra truth tests, properties, or repr(user_object).
Every executed CPython opcode in the declared roots is recorded. Values are
frame locals at opcode boundaries, NOT access to the private operand stack.
Native calls and GPU work keep their independent dispatch/completion boundary.
"""
from __future__ import annotations
from xiaomai_observe.passive import type_is_one_of, instance_dict
from xiaomai_python.core import Array as _ReferenceArray
from xiaomai_observe.code_identity import code_fingerprint
import asyncio
import contextlib
import collections
import dis
import hashlib
import marshal
import struct
from pathlib import Path
import sys
import types
import threading
import weakref
from .capture import BUSY,measurement
from .passive import actual_isinstance, qualified_type, exception_info
from .journal import EvidenceError,digest
from .local_snapshots import LocalSnapshots, variable_fingerprints, diff_names
from .runtime import LineTracer,_SCOPE
import torch

class InstructionTracer(LineTracer):
    def __init__(self,monitor,roots,*,capture_locals=True,require_roots=True):
        super().__init__(monitor,roots,capture_locals=capture_locals)
        if require_roots and not self.roots:raise EvidenceError('EMPTY_PYTHON_TRACE_SCOPE')
        self.compact_locals=monitor.options.get('compact_locals',False)
        self.value_transitions=monitor.options.get('value_transitions',False)
        self.native_calls=monitor.options.get('native_calls',False)
        self.snapshots=LocalSnapshots(monitor.journal);self.callee_calls={};self.callee_counter=0
        self.instructions={};self.code_ids={};self.dynamic={};self.events=0;self.edges=0
        self.task_ids=weakref.WeakKeyDictionary();self.next_task=0
        self.unresolved=set();self.suspended=set();self.active_threads=set()
        self.opcode_inventory={};self._eligible={};self._activation_count=0;self._monitoring_id=None;self._monitoring_callbacks={};self._activation_lock=threading.RLock();self.implicit_python_threads=set()
    def register_dynamic(self,code,source):
        """Admit actual compiled source only when its recursively compiled code matches."""
        if not actual_isinstance(code,types.CodeType) or type(source) is not str:
            raise TypeError('CODE_AND_SOURCE_REQUIRED')
        compiled=compile(source,code.co_filename,'exec',dont_inherit=True)
        def walk(c):
            yield c
            for x in c.co_consts:
                if actual_isinstance(x,types.CodeType):yield from walk(x)
        wanted=code_fingerprint(code)
        matches=[c for c in walk(compiled) if code_fingerprint(c)==wanted]
        if not matches:raise EvidenceError('DYNAMIC_SOURCE_CODE_MISMATCH')
        raw=source.encode('utf-8');self.dynamic[code]=(source,digest(raw));self._eligible[code]=True
        self.m.journal.emit('DYNAMIC_SOURCE',file=code.co_filename,content=self.m.journal.blob(raw),loaded_code_object_sha256=wanted)
    def task_id(self):
        try:t=asyncio.current_task()
        except RuntimeError:return None
        if t is None:return None
        if t not in self.task_ids:
            self.task_ids[t]=f'task:{self.next_task}';self.next_task+=1
        return self.task_ids[t]
    def included_code(self,code):
        if code in self.dynamic:return True
        if code not in self._eligible:self._eligible[code]=not code.co_filename.startswith('<') and self.included(code.co_filename)
        return self._eligible[code]
    def safe(self,value,path,ancestors=None):
        """Capture builtins by exact type; unknown objects are references, not inspected."""
        ancestors=set() if ancestors is None else ancestors
        ty=type(value)
        if ty is float and self.compact_locals:
            return {'kind':'float64','value':self.m.capture.tree(value,path),'bits_le':struct.pack('<d',value).hex()}
        if type_is_one_of(ty,(type(None),bool,int,float,str,bytes)):return self.m.capture.tree(value,path)
        if type_is_one_of(ty,(torch.Tensor,torch.nn.Parameter)):return self.m.capture.tensor(value,path)
        if id(value) in ancestors:return {'kind':'cycle','object_id':id(value)}
        anc=ancestors|{id(value)}
        if type_is_one_of(ty,(dict,collections.OrderedDict)):
            # Numeric path segments avoid str/repr of arbitrary mapping keys.
            return {'kind':'mapping','items':[[self.safe(k,f'{path}.key[{i}]',anc),self.safe(v,f'{path}.value[{i}]',anc)] for i,(k,v) in enumerate(value.items())]}
        if type_is_one_of(ty,(tuple,list)):return {'kind':ty.__name__,'items':[self.safe(v,f'{path}[{i}]',anc) for i,v in enumerate(value)]}
        if ty is _ReferenceArray:
            attrs=instance_dict(value)
            return {'kind':'reference_array','shape':list(attrs['shape']),'data':self.safe(attrs['data'],path+'.data',anc)}
        if actual_isinstance(value,torch.nn.Module):return {'kind':'module_reference','object_id':id(value),'type':qualified_type(ty)}
        if type_is_one_of(ty,(torch.dtype,torch.device)):return {'kind':ty.__name__,'value':str(value)}
        key=(path,qualified_type(ty))
        if key not in self.unresolved:
            self.unresolved.add(key);self.m.capture.gap('INSTRUCTION_LOCAL_OPAQUE_REFERENCE',path=path,type=qualified_type(ty))
        return {'kind':'opaque_reference','object_id':id(value),'type':qualified_type(ty),'value_capture':'NOT_INSPECTED_NO_DESCRIPTOR_EXECUTION'}
    def __call__(self,frame,event,arg,offset=None,terminal=None):
        if BUSY.get():return self
        code=frame.f_code
        if not self.included_code(code):return None
        with measurement():
            self.events+=1
            key=id(frame)
            if code not in self.instructions:
                self.instructions[code]={i.offset:i for i in dis.get_instructions(code,show_caches=True)}
                self.code_ids[code]=code_fingerprint(code)
            actual_offset=frame.f_lasti if offset is None else offset
            ops=self.instructions[code];at=ops.get(actual_offset)
            if key not in self.frames:
                self.frames[key]={'id':self.m.new_id('python'),'last_opcode':None,'exception_seen':False}
            state=self.frames[key]
            resumed=event=='call' and key in self.suspended
            if event=='call':
                if self._monitoring_id is None:frame.f_trace_lines=True;frame.f_trace_opcodes=True
                self.suspended.discard(key)
            filename=code.co_filename
            if code in self.dynamic:
                source_hash=self.dynamic[code][1]
            else:
                if filename not in self.files:
                    try:
                        raw=Path(filename).read_bytes();ref=self.m.journal.blob(raw)
                        self.files[filename]=ref['sha256'];self.m.journal.emit('PYTHON_SOURCE',file=filename,content=ref)
                    except OSError as exc:
                        self.files[filename]=None;self.m.capture.gap('SOURCE_READ_FAILED',file=filename,error=type(exc).__name__)
                source_hash=self.files[filename]
            parent=frame.f_back
            while parent is not None and id(parent) not in self.frames:parent=parent.f_back
            parent_id=self.frames[id(parent)]['id'] if parent is not None else None
            payload=dict(call_id=state['id'],parent_call_id=parent_id,parent_scope=list(_SCOPE.get()),
                         file=filename,line=frame.f_lineno,qualname=code.co_qualname,
                         source_sha256=source_hash,loaded_code_object_sha256=self.code_ids[code],
                         instruction_offset=actual_offset,task_id=self.task_id(),resumed=resumed)
            self.observed.add((filename,frame.f_lineno))
            if at is not None:
                pos=at.positions
                payload.update(opcode_name=at.opname,opcode_number=at.opcode,
                               line_start=pos.lineno,line_end=pos.end_lineno,column_start=pos.col_offset,column_end=pos.end_col_offset)
            if self.capture_locals:
                safe_locals=self.safe(dict(frame.f_locals),'locals')
                payload['locals']=self.snapshots.put(safe_locals) if self.compact_locals else safe_locals
                if self.value_transitions and event in ('opcode','return','exception'):
                    ref=payload['locals'] if self.compact_locals else self.snapshots.put(safe_locals)
                    fingerprints=variable_fingerprints(safe_locals)
                    previous=state.get('value_boundary')
                    if previous is not None:
                        old_offset,old_ref,old_fingerprints=previous
                        self.m.journal.emit('PYTHON_VALUE_TRANSITION',**{k:v for k,v in payload.items() if k!='locals'},
                            from_offset=old_offset,to_offset=actual_offset,before=old_ref,after=ref,
                            variables=diff_names(old_fingerprints,fingerprints),event_at_end=event,
                            attribution='OBSERVED_BOUNDARY_INTERVAL_MAY_INCLUDE_CHILD_CALLS_NOT_SEMANTIC_CAUSAL_PROOF')
                    state['value_boundary']=(actual_offset,ref,fingerprints)
                payload['value_boundary']='BEFORE_CURRENT_OPCODE_OR_AT_RUNTIME_EVENT_NOT_OPERAND_STACK'
            if event=='opcode':
                prev=state['last_opcode']
                if prev is not None:
                    pi=ops.get(prev);jump=pi is not None and pi.opcode in set(dis.hasjrel+dis.hasjabs)
                    self.m.journal.emit('PYTHON_INSTRUCTION_EDGE',**payload,from_offset=prev,to_offset=actual_offset,
                        conditional_or_jump=jump,jump_target=pi.argval if jump else None,
                        jumped_to_target=(actual_offset==pi.argval) if jump else None,
                        boundary='NEXT_OBSERVED_OPCODE; INTERVENING_CHILD_CALLS_HAVE_OWN_EVENTS')
                    self.edges+=1
                state['last_opcode']=actual_offset
                k=(self.code_ids[code],actual_offset);self.opcode_inventory[k]=self.opcode_inventory.get(k,0)+1
            if event=='exception':
                state['exception_seen']=True
                payload['exception_type']=exception_info(arg[1])['type']
                payload['exception_object_id']=id(arg[1])
                # Do not execute arbitrary exception __str__ or repr.
                state['last_opcode']=None
            if event=='return':
                opname=at.opname if at else None
                if terminal=='yield':opname='YIELD_VALUE'
                if terminal=='returned':opname='RETURN_VALUE'
                if terminal=='unwind':opname='EXCEPTION_UNWIND'
                payload['opcode_at_event']=opname
                payload['return_or_yield']=self.safe(arg,'return')
                payload['exception_seen_in_frame']=state['exception_seen']
                payload['terminal_classification']='SUSPEND' if opname in ('YIELD_VALUE','YIELD_FROM') else 'RETURN_VALUE' if opname in ('RETURN_VALUE','RETURN_CONST') else 'UNWIND_OR_NONRETURN_RUNTIME_EVENT'
                if opname in ('YIELD_VALUE','YIELD_FROM'):
                    self.suspended.add(key);state['last_opcode']=None
                else:self.frames.pop(key,None);self.suspended.discard(key)
            self.m.journal.emit('PYTHON_'+event.upper(),**payload)
        return self
    def _monitor_callback(self,name):
        def callback(code,position,*args):
            if BUSY.get() or not self.included_code(code):return None
            tid=threading.get_ident()
            if tid not in self.m.thread_ids and tid not in self.implicit_python_threads:
                self.implicit_python_threads.add(tid)
                with measurement():self.m.capture.gap('PYTHON_THREAD_OBSERVED_ATEN_MODE_NOT_ENROLLED',thread_id=tid)
            frame=sys._getframe(1)
            while frame is not None and frame.f_code is not code:frame=frame.f_back
            if frame is None:
                with measurement():self.m.capture.gap('MONITOR_FRAME_NOT_FOUND',event=name,qualname=code.co_qualname)
                return None
            if name in ('CALL','C_RETURN','C_RAISE'):
                with measurement():
                    callable_obj,arg0=args
                    ty=type(callable_obj)
                    if type_is_one_of(ty,(types.FunctionType,types.BuiltinFunctionType,types.MethodType,types.BuiltinMethodType)):
                        description={'type':qualified_type(ty),'module':callable_obj.__module__,
                            'qualname':callable_obj.__qualname__,'object_id':id(callable_obj)}
                        f=callable_obj.__func__ if ty is types.MethodType else callable_obj
                        if type(f) is types.FunctionType:description['code_sha256']=code_fingerprint(f.__code__)
                    else:
                        description={'type':qualified_type(ty),'object_id':id(callable_obj),
                                     'attributes':'NOT_INSPECTED'}
                    key=(id(frame),position)
                    if name=='CALL':
                        ident='callee:'+str(self.callee_counter);self.callee_counter+=1
                        # Python call completion has its own PY_RETURN/PY_UNWIND.
                        if not type_is_one_of(ty,(types.FunctionType,types.MethodType)):self.callee_calls[key]=ident
                    else:ident=self.callee_calls.pop(key,None)
                    self.m.journal.emit('PYTHON_CALLEE_'+name,call_id=self.frames.get(id(frame),{}).get('id'),
                        callee_call_id=ident,callee=description,source_file=code.co_filename,
                        instruction_offset=position,task_id=self.task_id(),
                        arg0_available=arg0 is not sys.monitoring.MISSING,
                        arg0=None if arg0 is sys.monitoring.MISSING else self.safe(arg0,'callee.arg0'),
                        remaining_arguments='NOT_EXPOSED_BY_MONITORING_API',
                        returned_value='NOT_EXPOSED_BY_C_RETURN_CALLBACK' if name=='C_RETURN' else None,
                        boundary='CPYTHON_CALL_PORT_NOT_NATIVE_INTERNAL_INSTRUCTIONS')
                return None
            if name in ('PY_START','PY_RESUME','PY_THROW'):
                self(frame,'call',None,position)
            elif name in ('PY_RETURN','PY_YIELD','PY_UNWIND'):
                self(frame,'return',args[0] if name!='PY_UNWIND' else None,position,
                     terminal={'PY_RETURN':'returned','PY_YIELD':'yield','PY_UNWIND':'unwind'}[name])
            elif name in ('RAISE','RERAISE'):
                exc=args[0];self(frame,'exception',(type(exc),exc,None),position)
            elif name=='INSTRUCTION':self(frame,'opcode',None,position)
            elif name=='LINE':self(frame,'line',None,frame.f_lasti)
            elif name=='EXCEPTION_HANDLED':self(frame,'exception_handled',None,position)
            else:
                # Runtime branch edges, unlike an inferred branch, include its
                # actual destination; the condition is NOT evaluated a second time.
                with measurement():
                    self.m.journal.emit('PYTHON_RUNTIME_BRANCH',file=code.co_filename,qualname=code.co_qualname,
                        instruction_offset=position,destination_offset=args[0],runtime_event=name,
                        call_id=self.frames.get(id(frame),{}).get('id'),task_id=self.task_id())
            return None
        return callback
    @contextlib.contextmanager
    def activate(self):
        if not hasattr(sys,'monitoring'):
            raise EvidenceError('INSTRUCTION_MONITORING_REQUIRES_CPYTHON_3_12_PLUS')
        mon=sys.monitoring
        with self._activation_lock:
            if self._activation_count==0:
                chosen=None
                for ident in (3,4,5,2,1,0):
                    try:mon.use_tool_id(ident,'xiaomai-instructions');chosen=ident;break
                    except ValueError:pass
                if chosen is None:raise EvidenceError('NO_FREE_MONITORING_TOOL_ID')
                self._monitoring_id=chosen
                names=['PY_START','PY_RESUME','PY_THROW','PY_RETURN','PY_YIELD','PY_UNWIND',
                       'RAISE','RERAISE','EXCEPTION_HANDLED','LINE','INSTRUCTION','JUMP']
                if self.native_calls:names+=['CALL','C_RETURN','C_RAISE']
                names+=['BRANCH'] if hasattr(mon.events,'BRANCH') else ['BRANCH_LEFT','BRANCH_RIGHT']
                self._event_mask=0
                try:
                    for name in names:
                        event=getattr(mon.events,name);cb=self._monitor_callback(name)
                        old=mon.register_callback(chosen,event,cb)
                        if old is not None:raise EvidenceError('UNEXPECTED_EXISTING_MONITORING_CALLBACK')
                        self._monitoring_callbacks[event]=cb;self._event_mask|=event
                    mon.set_events(chosen,self._event_mask)
                    # CPython may omit ancillary C_RETURN/C_RAISE bits from
                    # get_events: they are controlled by CALL. Accept only this
                    # documented normalization, never arbitrary missing events.
                    ancillary=mon.events.C_RETURN|mon.events.C_RAISE
                    self._effective_event_mask=mon.get_events(chosen)
                    if self._effective_event_mask not in (self._event_mask,self._event_mask & ~ancillary):
                        raise EvidenceError('UNEXPECTED_MONITORING_EVENT_NORMALIZATION')
                    with measurement():
                        self.m.journal.emit('PYTHON_MONITORING_POLICY',requested_event_mask=self._event_mask,
                            effective_event_mask=self._effective_event_mask,ancillary_event_mask=ancillary,
                            normalization='C_RETURN_C_RAISE_CONTROLLED_BY_CALL')
                except BaseException:
                    mon.set_events(chosen,0)
                    for event in self._monitoring_callbacks:mon.register_callback(chosen,event,None)
                    mon.free_tool_id(chosen);self._monitoring_id=None
                    raise
            self._activation_count+=1
        try:yield
        finally:
            with self._activation_lock:
                self._activation_count-=1
                if self._activation_count==0:
                    ident=self._monitoring_id
                    drift=mon.get_tool(ident)!='xiaomai-instructions' or mon.get_events(ident)!=self._effective_event_mask
                    # Unregister only our own slot, never disable another tool.
                    if mon.get_tool(ident)=='xiaomai-instructions':
                        mon.set_events(ident,0)
                        for event,cb in self._monitoring_callbacks.items():
                            old=mon.register_callback(ident,event,None)
                            drift=drift or old is not cb
                        mon.free_tool_id(ident)
                    self._monitoring_id=None
                    if drift:self.m.capture.gap('PYTHON_MONITORING_CHANGED_DURING_RUN')
    def coverage(self):
        return dict(engine='sys.monitoring',local_snapshot_storage=self.snapshots.stats(),native_call_ports_enabled=self.native_calls,unmatched_native_calls=len(self.callee_calls),implicit_python_threads=sorted(self.implicit_python_threads),instruction_events=self.events,observed_edges=self.edges,async_tasks_assigned=self.next_task,
                    distinct_instruction_locations=len(self.opcode_inventory),
                    instruction_invocations=sum(self.opcode_inventory.values()),
                    suspended_frames=len(self.suspended),dynamic_sources=len(self.dynamic),
                    empty_scope=not bool(self.roots or self.dynamic),operand_stack_values=False,
                    statement_value_contract='LOCALS_AT_OPCODE_BOUNDARIES_PLUS_ATEN_INPUT_OUTPUT',
                    native_instruction_coverage=False)
