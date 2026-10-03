"""Stable structural/value fingerprint of a CPython code definition.

marshal is a serialization with internal reference encodings; its byte stream is
not a canonical source identity. This fingerprint binds executable fields and
constant VALUES (with exact float bits), not outside object identity, globals,
closure contents, library state, or a claim of semantic equivalence. Those are
separate source/state contracts. No repr, properties, or arbitrary codecs run.
"""
from __future__ import annotations
import hashlib
import json
import struct
import types

SCHEMA='CPYTHON_CODE_STRUCTURAL_VALUES_V1'

def code_descriptor(code):
    if type(code) is not types.CodeType:raise TypeError('EXACT_PYTHON_CODE_REQUIRED')
    def constant(x):
        t=type(x)
        if x is None:return {'kind':'none'}
        if x is Ellipsis:return {'kind':'ellipsis'}
        if t is bool:return {'kind':'bool','value':x}
        if t is int:return {'kind':'int','value':str(x)}
        if t is str:return {'kind':'str','value':x}
        if t is bytes:return {'kind':'bytes','hex':x.hex()}
        if t is float:return {'kind':'float64','bits_le':struct.pack('<d',x).hex()}
        if t is complex:return {'kind':'complex128','bits_le':struct.pack('<dd',x.real,x.imag).hex()}
        if t is tuple:return {'kind':'tuple','items':[constant(v) for v in x]}
        if t is frozenset:
            items=[constant(v) for v in x]
            items.sort(key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')))
            return {'kind':'frozenset','items':items}
        if t is types.CodeType:return {'kind':'code','value':code_descriptor(x)}
        raise TypeError('UNSUPPORTED_CODE_CONSTANT:'+t.__module__+'.'+t.__qualname__)
    fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals',
            'co_stacksize','co_flags','co_filename','co_name','co_qualname',
            'co_firstlineno','co_names','co_varnames','co_freevars','co_cellvars')
    d={name:getattr(code,name) for name in fields}
    d['co_code']=code.co_code.hex();d['co_linetable']=code.co_linetable.hex()
    d['co_exceptiontable']=getattr(code,'co_exceptiontable',b'').hex()
    d['constants']=[constant(x) for x in code.co_consts]
    return {'schema':SCHEMA,**d}

def code_fingerprint(code):
    raw=json.dumps(code_descriptor(code),ensure_ascii=True,sort_keys=True,separators=(',',':')).encode('ascii')
    return hashlib.sha256(raw).hexdigest()
