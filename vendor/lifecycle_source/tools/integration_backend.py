"""Load the locally supplied candidate dispatch over explicit reference support.

This is a mixed-version laboratory assembly, not an original True checkpoint.
V083 core/dispatch/index source identity and every support file are recorded.
"""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[1]
SUPPORT=ROOT/'reference_support'
def load():
    for root in (ROOT,SUPPORT):
        if str(root) not in sys.path:sys.path.insert(0,str(root))
    import xiaomai_python
    if not SUPPORT.is_dir():
        from xiaomai_python import dispatch
        if not hasattr(dispatch,'PYTHON_STORAGE_OPERATORS'):raise RuntimeError('CANDIDATE_BACKEND_NOT_INSTALLED')
        return dispatch
    name='xiaomai_python.dispatch_storage_v085'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,ROOT/'integration/dispatch.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
