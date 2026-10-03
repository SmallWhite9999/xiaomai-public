"""Auditable Python arithmetic backend and native-state engineering.

Storage, tensor construction, autograd scheduling and declared primitives remain
external. Strict dispatch never silently treats an unimplemented kernel as Python.
"""
__version__='0.8.2'
