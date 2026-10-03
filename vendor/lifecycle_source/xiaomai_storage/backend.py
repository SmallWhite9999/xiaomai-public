"""Interpose storage kernels on the same live tensor calls, without a second model."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from . import kernels,index_ops,view_ops,numeric_words,placement_ops,sequence_ops,bridge_ops,allocation
ALL_OPERATORS=kernels.OPERATORS|index_ops.OPERATORS|view_ops.OPERATORS|numeric_words.OPERATORS|placement_ops.OPERATORS|sequence_ops.OPERATORS|bridge_ops.OPERATORS|allocation.OPS

def schema_values(func,args,kwargs):
    result=dict(kwargs)
    for spec,value in zip(func._schema.arguments,args):
        if spec.name in result:raise kernels.StorageError("DUPLICATE_ARGUMENT")
        result[spec.name]=value
    for spec in func._schema.arguments:
        if spec.name not in result and spec.has_default_value():
            result[spec.name]=spec.default_value
    return result

def descriptor(value):
    if type(value) is torch.Tensor or type(value) is torch.nn.Parameter:
        kernels.admit(value)
        return dict(type="tensor",shape=list(value.shape),stride=list(value.stride()),
                    storage_offset=value.storage_offset(),dtype=str(value.dtype),
                    object_id=id(value),storage_address=value.untyped_storage().data_ptr(),
                    version=None if value.is_inference() else value._version)
    if any(type(value) is t for t in (int,float,bool,str)) or value is None:
        if type(value) is float:return dict(type="float",hex=value.hex())
        return value
    if any(type(value) is t for t in (tuple,list)):return [descriptor(x) for x in value]
    return {"opaque":True,"type_id":id(type(value))}

class EventFile:
    """Synchronous append-only journal; write failure propagates, no sampling."""
    def __init__(self,path,*,fsync_each=False):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open("x",encoding="utf-8",newline="\n")
        self.sequence=0;self.previous="0"*64;self.lock=threading.Lock()
        self.fsync_each=fsync_each
        self.closed=False
    def emit(self,kind,**fields):
        with self.lock:
            if self.closed:raise RuntimeError("JOURNAL_CLOSED")
            row=dict(sequence=self.sequence,previous_sha256=self.previous,
                     kind=kind,pid=os.getpid(),thread_id=threading.get_ident(),**fields)
            encoded=json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)
            digest=hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            self.stream.write(json.dumps(dict(row,sha256=digest),sort_keys=True,
                                         separators=(",",":"),allow_nan=False)+"\n")
            self.stream.flush()
            if self.fsync_each:os.fsync(self.stream.fileno())
            self.previous=digest;self.sequence+=1
    def close(self):
        if not self.closed:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):
        try:self.emit("JOURNAL_END",execution="RETURNED" if exc_type is None else "RAISED")
        finally:self.close()

def verify_event_file(path):
    previous="0"*64;calls={};ended=set();policies=0;closed=0;events=0;pending={};last_kind=None;journal_end=0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):raise ValueError("TRUNCATED_LINE")
            row=json.loads(line);digest=row.pop("sha256")
            if row["sequence"]!=events or row["previous_sha256"]!=previous:
                raise ValueError("EVENT_SEQUENCE_OR_CHAIN")
            actual=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),
                                            allow_nan=False).encode()).hexdigest()
            if digest!=actual:raise ValueError("EVENT_HASH")
            if row["kind"]=="STORAGE_POLICY":policies+=1
            if row["kind"]=="STORAGE_BACKEND_CLOSED":
                closed+=1
                if row["failed"]:raise ValueError("BACKEND_FAILED")
            key=row.get("call_id")
            last_kind=row["kind"]
            if row["kind"]=="JOURNAL_END":
                journal_end+=1
                if row["execution"]!="RETURNED":raise ValueError("JOURNAL_FAILED")
            if row["kind"]=="STORAGE_WRITE_INTENT":
                write_key=(key,row["dst_byte"])
                if key not in calls or write_key in pending:raise ValueError("WRITE_NOT_PAIRED")
                pending[write_key]=len(bytes.fromhex(row["after_hex"]))
            if row["kind"]=="STORAGE_WRITE_DONE":
                write_key=(key,row["dst_byte"])
                if pending.pop(write_key,None)!=row["byte_count"]:raise ValueError("WRITE_NOT_PAIRED")
            if row["kind"]=="STORAGE_CALL":
                if key in calls:raise ValueError("DUPLICATE_CALL")
                calls[key]=row
            if row["kind"] in ("STORAGE_RETURN","STORAGE_ERROR"):
                if key not in calls or key in ended:raise ValueError("UNPAIRED_END")
                ended.add(key)
                if row["kind"]=="STORAGE_ERROR":raise ValueError("EXECUTION_ERROR")
            previous=digest;events+=1
    if policies!=1 or closed!=1 or set(calls)!=ended or pending or journal_end!=1 or last_kind!="JOURNAL_END":raise ValueError("INCOMPLETE_JOURNAL")
    return {"events":events,"calls":len(calls),"chain_sha256":previous,
            "scope":"Only this extension's dispatch calls; not global coverage"}

class StorageBackend(TorchDispatchMode):
    """Nest inside PythonReferenceBackend to extend existing caller-owned execution.

    Unsupported inputs to a selected operation are rejected. Other operations
    delegate unchanged to the next mode; delegation is explicitly reported.
    """
    def __init__(self,*,sink=None,progress=None,record_values=False):
        super().__init__()
        self.sink=sink;self.progress=progress;self.record_values=record_values
        self.calls=[];self.failed=False;self.ordinal=0
        self.owner_thread=None
        self.identity="storage:"+str(id(self))
    def emit(self,kind,**fields):
        if self.sink is not None:self.sink(kind,**fields)
    def __enter__(self):
        self.owner_thread=threading.get_ident()
        self.emit("STORAGE_POLICY",operators=sorted(ALL_OPERATORS),
                  values="ALL_WRITTEN_BYTES" if self.record_values else "METADATA_ONLY",
                  scope="CURRENT_THREAD_DISPATCH",
                  native_dependencies=["allocation","metadata","scalar_dtype_conversion",
                                       "CPython_raw_memory","autograd_scheduling"],
                  unsupported_selected_operation="REJECT",
                  unrelated_operations="DELEGATE_UNCHANGED")
        return super().__enter__()
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:
            self.emit("STORAGE_BACKEND_CLOSED",failed=self.failed or args[0] is not None,
                      calls=len(self.calls),entire_true_verified=False)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=str(func);ordinal=self.ordinal;self.ordinal+=1
        call_id=self.identity+":"+str(ordinal)
        implemented=name in ALL_OPERATORS
        location=None
        frame=inspect.currentframe().f_back
        while frame is not None:
            filename=frame.f_code.co_filename.replace("\\","/")
            if "/torch/" not in filename and "/xiaomai_storage/" not in filename:
                location=dict(file=frame.f_code.co_filename,line=frame.f_lineno,
                              function=frame.f_code.co_name,bytecode_offset=frame.f_lasti)
                break
            frame=frame.f_back
        del frame
        row=dict(call_id=call_id,operator=name,source=location,
                 implementation=("PYTHON_ALLOCATION_NATIVE_BINDING" if name in allocation.OPS else "PYTHON_VIEW_PLAN_NATIVE_ALIAS" if name in view_ops.OPERATORS|bridge_ops.VIEW_OPERATORS else "PYTHON_STORAGE") if implemented else "DELEGATED_UNCHANGED")
        self.calls.append(row)
        self.emit("STORAGE_CALL",**row)
        try:
            if not implemented:
                result=func(*args,**kwargs)
            else:
                values=schema_values(func,args,kwargs)
                self.emit("STORAGE_ARGUMENTS",**row,arguments={k:descriptor(v) for k,v in values.items()})
                def events(kind,**fields):self.emit(kind,call_id=call_id,**fields)
                # This bypass covers native allocation/metadata only. No selected
                # data kernel is invoked from kernels.execute().
                with torch._C._DisableTorchDispatch(),torch.no_grad():
                    executor=allocation.execute if name in allocation.OPS else bridge_ops.execute if name in bridge_ops.OPERATORS else placement_ops.execute if name in placement_ops.OPERATORS else sequence_ops.execute if name in sequence_ops.OPERATORS else numeric_words.execute if name in numeric_words.OPERATORS else view_ops.execute if name in view_ops.OPERATORS else index_ops.execute if name in index_ops.OPERATORS else kernels.execute
                    result=executor(name,values,
                        sink=events if self.record_values else None,progress=self.progress)
            self.emit("STORAGE_RETURN",**row)
            return result
        except BaseException as exc:
            self.failed=True
            self.emit("STORAGE_ERROR",**row,error_type=type(exc).__name__)
            raise

def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()

PINS={
    "xiaomai_python/dispatch.py":"5dbc309235e9dcc2d22e049a6714a14342c20869",
    "xiaomai_python/native_ops.py":"a9d6160efda81c58f126875db3bab002ddb54094",
}

@contextmanager
def integrated_backend(*,journal=None,progress=None,record_values=False,**kwargs):
    """Use already-imported pinned v0.8.3 backend. Does not install/replace weights."""
    from xiaomai_python import dispatch,native_ops
    sources={
        "xiaomai_python/dispatch.py":Path(dispatch.__file__),
        "xiaomai_python/native_ops.py":Path(native_ops.__file__),
    }
    for name,path in sources.items():
        if git_blob(path.read_bytes())!=PINS[name]:
            raise kernels.StorageError("BASELINE_SOURCE_MISMATCH:"+name)
    sink=journal.emit if journal is not None else None
    with dispatch.PythonReferenceBackend(journal=journal,progress=progress,**kwargs) as arithmetic:
        progress_callback=(lambda **f:progress.work(**f)) if progress is not None else None
        with StorageBackend(sink=sink,progress=progress_callback,record_values=record_values) as storage:
            yield {"arithmetic":arithmetic,"storage":storage,
                   "base_commit":"ccf7b740a90d8fc3611a56e10d05b2486e156984"}
