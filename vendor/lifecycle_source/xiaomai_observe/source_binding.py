"""Match loaded Python code objects to current file definitions without executing files.

This is bytecode/constant binding in THIS interpreter. It does not establish the
identity of every global, closure value, C extension, decorator, or GPU kernel.
Text provenance cannot be reconstructed from an arbitrary already-loaded object.
"""
from __future__ import annotations
from xiaomai_observe.passive import type_is_one_of
from xiaomai_observe.code_identity import code_fingerprint
import hashlib
import json
import marshal
import struct
from pathlib import Path
import sys
import types
from .journal import canonical, digest


def _constant(value):
    if isinstance(value, types.CodeType): return {'code': executable_identity(value)}
    if value is None or type_is_one_of(type(value),(bool, int, str)): return value
    if type(value) is bytes: return {'bytes':value.hex()}
    if type(value) is float: return {'float_bits_le':struct.pack('<d',value).hex()}
    if type(value) is complex: return {'complex_bits_le':struct.pack('<dd',value.real,value.imag).hex()}
    if type(value) is tuple: return {'tuple':[_constant(v) for v in value]}
    if type(value) is frozenset:
        rows=[_constant(v) for v in value]
        return {'frozenset':sorted(rows,key=canonical)}
    if value is Ellipsis: return {'ellipsis':True}
    # Python code constants use a bounded type family, not arbitrary repr hooks.
    raise TypeError('UNSUPPORTED_CODE_CONSTANT: '+type(value).__qualname__)


def executable_identity(code):
    row={'bytecode':code.co_code.hex(),'consts':[_constant(v) for v in code.co_consts],
         'names':code.co_names,'varnames':code.co_varnames,'freevars':code.co_freevars,
         'cellvars':code.co_cellvars,'flags':code.co_flags,'argcount':code.co_argcount,
         'posonlyargcount':code.co_posonlyargcount,'kwonlyargcount':code.co_kwonlyargcount,
         'exceptiontable':getattr(code,'co_exceptiontable',b'').hex()}
    return digest(canonical(row))


def _walk(code):
    yield code
    for value in code.co_consts:
        if isinstance(value,types.CodeType): yield from _walk(value)


class SourceAudit:
    def __init__(self):
        self.files={};self.compiled={}

    def probe(self,fn):
        f=getattr(fn,'__func__',fn);code=getattr(f,'__code__',None)
        if code is None:
            return {'status':'NO_PYTHON_CODE_OBJECT','native_internals':'NOT_OBSERVED'}
        result={'file':code.co_filename,'loaded_qualname':code.co_qualname,
                'loaded_first_line':code.co_firstlineno,
                'loaded_code_object_sha256':code_fingerprint(code),'code_identity_schema':'CPYTHON_CODE_STRUCTURAL_VALUES_V1',
                'globals_and_closure_values_verified':False}
        path=Path(code.co_filename)
        try:
            if code.co_filename.startswith('<'): raise OSError('synthetic filename')
            raw=path.read_bytes();sha=digest(raw)
            previous=self.files.get(str(path))
            self.files.setdefault(str(path),sha)
            result['source_file_sha256']=sha
            if previous is not None and sha!=previous:
                result['source_changed_during_inventory']=True
            cache_key=(str(path),sha)
            if cache_key not in self.compiled:
                compiled=compile(raw,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
                self.compiled[cache_key]=list(_walk(compiled))
            executable=executable_identity(code)
            result['loaded_executable_sha256']=executable
            candidates=[c for c in self.compiled[cache_key] if c.co_qualname==code.co_qualname]
            matches=[c for c in candidates if executable_identity(c)==executable]
            result['disk_candidate_lines']=[c.co_firstlineno for c in candidates]
            result['matched_disk_lines']=[c.co_firstlineno for c in matches]
            result['status']='EXECUTABLE_MATCH' if matches else 'LOADED_CODE_DISAGREES_WITH_DISK'
            result['same_line_mapping']=any(c.co_firstlineno==code.co_firstlineno and
                c.co_linetable==code.co_linetable for c in matches)
            result['source_at_load_time_verified']=False
        except (OSError,SyntaxError,ValueError,TypeError) as exc:
            result.update(status='SOURCE_BINDING_UNRESOLVED',reason=type(exc).__name__)
        return result

    def model(self,model):
        result=[]
        for name,module in model.named_modules(remove_duplicate=False):
            result.append({'name':name,'object_id':id(module),'binding':self.probe(module.forward)})
        return {'modules':result,'scope':'REGISTERED_FORWARD_PYTHON_CODE_ONLY',
                'compiler_python':sys.version,'optimize':sys.flags.optimize,
                'full_dependency_binding_verified':False}

    def recheck_files(self):
        rows=[]
        for path,expected in self.files.items():
            try:
                actual=digest(Path(path).read_bytes())
                status='UNCHANGED' if expected==actual else 'CHANGED_DURING_OBSERVATION'
            except OSError:
                actual=None;status='UNREADABLE_AT_END'
            rows.append({'path':path,'before_sha256':expected,'after_sha256':actual,'status':status})
        return rows
