"""Passive type classification: never consult value.__class__ or user repr.

An opaque value remains opaque. This module does not resolve arbitrary user
properties, virtual ABC registrations, tensor subclasses, or extension slots.
"""
from __future__ import annotations
import types

_TYPE_DICT = type.__dict__['__dict__']
_TYPE_MRO = type.__dict__['__mro__']
_TYPE_NAME = type.__dict__['__name__']
_TYPE_QUALNAME = type.__dict__['__qualname__']
_TYPE_MODULE = type.__dict__['__module__']
_MISSING = object()


def mro(cls):
    return _TYPE_MRO.__get__(cls, type(cls))


def class_dict(cls):
    return _TYPE_DICT.__get__(cls, type(cls))


def actual_isinstance(value, classes):
    """Only real inheritance, not spoofed __class__/custom __instancecheck__."""
    if type(classes) is tuple:
        return any(actual_isinstance(value, cls) for cls in classes)
    return any(base is classes for base in mro(type(value)))


def qualified_type(cls):
    namespace = class_dict(cls)
    module = _TYPE_MODULE.__get__(cls, type(cls))
    if type(module) is not str:
        module = '<nonliteral-module>'
    name = _TYPE_QUALNAME.__get__(cls, type(cls))
    if type(name) is not str:
        name = '<nonliteral-qualname>'
    return module + '.' + name


def class_attr(cls, name, default=None):
    """Unresolved class dictionary value, never descriptor execution."""
    for base in mro(cls):
        namespace = class_dict(base)
        if name in namespace:
            return namespace[name]
    return default


def instance_dict(value):
    """Read only the builtin __dict__ slot, reject shadowing properties.

    Not a generic getset resolver. Callers admit object classes separately.
    """
    if type(value) is types.SimpleNamespace:
        return types.SimpleNamespace.__dict__['__dict__'].__get__(value, type(value))
    descriptor = class_attr(type(value), '__dict__', _MISSING)
    if type(descriptor) is not types.GetSetDescriptorType:
        raise TypeError('NONSTANDARD_OBJECT_DICT_DESCRIPTOR')
    if descriptor.__name__ != '__dict__':
        raise TypeError('NONSTANDARD_OBJECT_DICT_DESCRIPTOR')
    result = descriptor.__get__(value, type(value))
    if type(result) is not dict:
        raise TypeError('NONSTANDARD_OBJECT_DICT')
    return result


def exception_info(exc):
    """Metadata without exception.__str__, custom args property or metaclass code.

    message is only the existing, exact string first/sole argument, not a call
    to the exception's formatter. Other arguments remain opaque references.
    """
    if not actual_isinstance(exc,BaseException):
        return {'type':qualified_type(type(exc)),'object_id':id(exc),'message':None,
                'message_semantics':'NOT_AN_EXCEPTION'}
    args=BaseException.__dict__['args'].__get__(exc,type(exc))
    return {'type':qualified_type(type(exc)),'object_id':id(exc),
            'message':args[0] if type(args) is tuple and len(args)==1 and type(args[0]) is str else None,
            'message_semantics':'EXACT_SOLE_STRING_ARG_NOT_EXCEPTION_STR',
            'arguments':[{'type':qualified_type(type(a)),'object_id':id(a)} for a in args]}


def type_is_one_of(cls, classes):
    """Identity comparisons only: never invoke a metaclass __eq__ or __hash__."""
    return any(cls is target for target in classes)


def static_instance_attr(value, name, default=_MISSING):
    """Resolve an instance field without inspect caches keyed by user classes.

    Return descriptors themselves rather than executing them. Builtin instance
    dictionaries are read only through the admitted __dict__ descriptor. A
    slots/property value remains a descriptor (and hence an explicit gap).
    """
    if type(name) is not str:
        raise TypeError('LITERAL_ATTRIBUTE_NAME_REQUIRED')
    declared = class_attr(type(value), name, _MISSING)
    if declared is not _MISSING and class_attr(type(declared), '__get__', _MISSING) is not _MISSING:
        if (class_attr(type(declared), '__set__', _MISSING) is not _MISSING or
                class_attr(type(declared), '__delete__', _MISSING) is not _MISSING):
            return declared
    try:
        values = instance_dict(value)
    except (TypeError, AttributeError):
        values = None
    if values is not None and name in values:
        return dict.__getitem__(values, name)
    if declared is not _MISSING:
        return declared
    if default is not _MISSING:
        return default
    raise AttributeError(name)
