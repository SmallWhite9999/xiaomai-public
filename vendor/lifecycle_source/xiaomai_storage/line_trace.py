"""No-sampling current-thread trace for explicit Python files.

Line/opcode observations, actual jump destinations and loaded-code identities are
recorded separately from source-byte identity. This does not expose C/GPU operand
stacks or arbitrarily evaluate user-defined values.
"""
from __future__ import annotations
import dis
import hashlib
from .code_identity import code_fingerprint,SCHEMA
from pathlib import Path
import struct
import sys
import threading

def literal_snapshot(value):
    t=type(value)
    if t is str or t is int or t is bool or value is None:return value
    if t is float:return {'float64_bits':struct.pack('<d',value).hex()}
    if t is bytes and len(value)<=65536:return {'bytes_hex':value.hex()}
    return {'unread_opaque_value':True,'type_id':id(t),'object_id':id(value)}

class LineTrace:
    def __init__(self,paths,sink,*,opcodes=True):
        self.paths={str(Path(p).resolve()) for p in paths}
        self.sink=sink;self.opcodes=opcodes;self.callback=self.trace
        self.events=0;self.branches=0;self.opcode_events=0;self.previous=None
        self.codes={};self.frames={};self.next_call=0
    def __enter__(self):
        if sys.gettrace() is not None:raise RuntimeError('EXISTING_TRACER_NOT_REPLACED')
        self.sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.paths}
        self.sink('PY_TRACE_POLICY',sources=self.sources,opcodes=self.opcodes,
                  sampling=False,scope='CURRENT_THREAD_EXPLICIT_FILES',
                  opaque_values='MARKED_NOT_EVALUATED',private_operand_stack=False,
                  branch_destination_observation=self.opcodes,code_identity='canonical executable fields and constant values; source equality is separate')
        self.owner=threading.get_ident()
        self.anchor=sys._getframe().f_back;self.anchor_opcode_flag=self.anchor.f_trace_opcodes
        if self.opcodes:self.anchor.f_trace_opcodes=True
        sys.settrace(self.callback);return self
    def code_record(self,code):
        key=id(code)
        if key not in self.codes:
            instructions={i.offset:i for i in dis.get_instructions(code)}
            try:identity={'sha256':code_fingerprint(code),'format':SCHEMA,'python':sys.version}
            except (TypeError,ValueError):identity={'unavailable':True,'reason':'unsupported_code_constant'}
            self.codes[key]=(code,instructions,identity)
            self.sink('PY_LOADED_CODE',file=code.co_filename,function=code.co_qualname,first_line=code.co_firstlineno,identity=identity,
                      source_sha256=self.sources[code.co_filename],source_equals_loaded_code_not_asserted=True)
        return self.codes[key]
    def trace(self,frame,event,arg):
        if frame.f_code.co_filename not in self.paths:return None
        # Reassert on the first line as well: CPython may not activate opcode
        # events for a newly entered code object from the call event alone.
        frame.f_trace_opcodes=self.opcodes
        _,instructions,identity=self.code_record(frame.f_code)
        key=id(frame)
        if event=='call':
            frame.f_trace_opcodes=self.opcodes
            self.frames[key]={'call':self.next_call,'previous_opcode':None};self.next_call+=1
        state=self.frames.setdefault(key,{'call':self.next_call,'previous_opcode':None})
        if event=='opcode':
            self.opcode_events+=1
            prior=state['previous_opcode']
            if prior is not None:
                instruction=instructions.get(prior)
                if instruction is not None and instruction.opcode in set(dis.hasjrel)|set(dis.hasjabs):
                    self.branches+=1
                    self.sink('PY_BRANCH_DESTINATION',file=frame.f_code.co_filename,function=frame.f_code.co_name,
                              frame_id=key,trace_call=state['call'],from_offset=prior,next_offset=frame.f_lasti,
                              opcode=instruction.opname,declared_target=instruction.argval,
                              arrived_at_target=frame.f_lasti==instruction.argval,
                              interpretation='actual bytecode destination, not semantic truth of a model claim')
            state['previous_opcode']=frame.f_lasti
        if event in ('call','line','opcode','return','exception'):
            self.events+=1
            snapshot={str(name):literal_snapshot(value) for name,value in frame.f_locals.items()}
            fields=dict(event=event,file=frame.f_code.co_filename,line=frame.f_lineno,
                        function=frame.f_code.co_name,frame_id=key,trace_call=state['call'],
                        bytecode_offset=frame.f_lasti,locals=snapshot)
            if event=='return':fields['returned']=literal_snapshot(arg)
            if event=='opcode':fields['opcode']=dis.opname[frame.f_code.co_code[frame.f_lasti]]
            self.sink('PY_EXECUTION',**fields)
        if event=='return':self.frames.pop(key,None)
        return self.callback
    def __exit__(self,exc_type,exc,tb):
        intact=sys.gettrace() is self.callback
        sys.settrace(None)
        self.anchor.f_trace_opcodes=self.anchor_opcode_flag;self.anchor=None
        same=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.sources.items())
        self.sink('PY_TRACE_CLOSED',events=self.events,branches=self.branches,opcode_events=self.opcode_events,
                  loaded_code_objects=len(self.codes),hook_intact=intact,source_unchanged=same,
                  execution='RETURNED' if exc_type is None else 'RAISED')
        if not intact or not same:raise RuntimeError('TRACE_HOOK_OR_SOURCE_CHANGED')
        if self.opcodes and self.events and not self.opcode_events:raise RuntimeError('REQUESTED_OPCODE_EVENTS_MISSING')
