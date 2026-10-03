"""Optional CPU compilation of explicitly written binary64 multiply-add loops.

Numba/LLVM and native ndarray storage are declared dependencies. fastmath=False,
no BLAS/dot/matmul call. This is NOT instruction-traced execution of each multiply.
Use the interpreted backend when every scalar step must be logged.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .kernels import StorageError,admit
from .backend import schema_values

try:
    import numba
except ImportError:
    numba=None

def multiply_loop(a,b):
    m,k=a.shape;n=b.shape[1]
    out=np.empty((m,n),dtype=np.float64)
    for i in range(m):
        for j in range(n):
            acc=0.0
            for t in range(k):
                product=np.float64(a[i,t])*np.float64(b[t,j])
                acc=acc+product
            out[i,j]=acc
    return out

def batch_loop(a,b):
    batch,m,k=a.shape;n=b.shape[2]
    out=np.empty((batch,m,n),dtype=np.float64)
    for z in range(batch):
        for i in range(m):
            for j in range(n):
                acc=0.0
                for t in range(k):
                    product=np.float64(a[z,i,t])*np.float64(b[z,t,j])
                    acc=acc+product
                out[z,i,j]=acc
    return out

def combine_loop(product,addend,alpha,beta):
    # addend is already a broadcast VIEW of the desired shape, not a copy.
    out=np.empty(product.shape,dtype=np.float64)
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            v=alpha*product[i,j]
            if beta!=0.0:v=v+beta*np.float64(addend[i,j])
            out[i,j]=v
    return out

_COMPILED={}
def compiled(which):
    if numba is None:raise StorageError("NUMBA_REQUIRED_NO_SILENT_NATIVE_FALLBACK")
    if which not in _COMPILED:
        fn={"mm":multiply_loop,"bmm":batch_loop,"combine":combine_loop}[which]
        _COMPILED[which]=numba.njit(fastmath=False,cache=True,nogil=True,boundscheck=True)(fn)
    return _COMPILED[which]

def array(t,rank=None):
    admit(t)
    if t.dtype is not torch.float32 and t.dtype is not torch.float64:
        raise StorageError("COMPILED_CPU_F32_F64_ONLY")
    if rank is not None and t.ndim!=rank:raise StorageError("COMPILED_RANK_MISMATCH")
    with torch._C._DisableTorchDispatch(),torch.no_grad():
        return t.detach().numpy()

OPS=frozenset(("aten.mm.default","aten.bmm.default","aten.addmm.default","aten.baddbmm.default"))

class CompiledMatmulBackend(TorchDispatchMode):
    def __init__(self,*,sink=None,full_scalar_trace=False):
        super().__init__()
        if full_scalar_trace:raise StorageError("COMPILED_MODE_CANNOT_CLAIM_FULL_SCALAR_TRACE")
        if numba is None:raise StorageError("NUMBA_REQUIRED_NO_SILENT_NATIVE_FALLBACK")
        self.sink=sink;self.calls=[];self.failed=False
    def emit(self,kind,**fields):
        if self.sink is not None:self.sink(kind,**fields)
    def __enter__(self):
        self.emit("COMPILED_CPU_POLICY",operators=sorted(OPS),fastmath=False,
                  native_dependencies=["Numba_LLVM","NumPy_storage","PyTorch_storage_dtype_cast","autograd"],
                  scalar_instruction_trace=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return super().__enter__()
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:self.emit("COMPILED_CPU_CLOSED",failed=self.failed or args[0] is not None,calls=len(self.calls))
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func);kwargs=kwargs or {}
        if name not in OPS:return func(*args,**kwargs)
        v=schema_values(func,args,kwargs);call={"operator":name,"ordinal":len(self.calls)}
        self.calls.append(call);started=time.perf_counter()
        self.emit("COMPILED_CPU_BEGIN",**call)
        try:
            batch=name in ("aten.bmm.default","aten.baddbmm.default")
            add=name in ("aten.addmm.default","aten.baddbmm.default")
            left=v.get("batch1",v.get("mat1",v.get("self")))
            right=v.get("batch2",v.get("mat2"))
            if left.dtype is not right.dtype:raise StorageError("COMPILED_DTYPE_MISMATCH")
            a,b=array(left,3 if batch else 2),array(right,3 if batch else 2)
            if a.shape[-1]!=b.shape[-2] or (batch and a.shape[0]!=b.shape[0]):
                raise StorageError("COMPILED_MATMUL_SHAPE")
            product=compiled("bmm" if batch else "mm")(a,b)
            if add:
                alpha,beta=v["alpha"],v["beta"]
                if type(alpha) not in (int,float) or type(beta) not in (int,float):
                    raise StorageError("COMPILED_REAL_ALPHA_BETA_REQUIRED")
                # A beta-zero addend is ignored, including its NaN/Inf values.
                if v["self"].dtype is not left.dtype:raise StorageError("COMPILED_ADDEND_DTYPE_MISMATCH")
                c=np.broadcast_to(array(v["self"]),product.shape)
                if batch:
                    for i in range(product.shape[0]):
                        product[i]=compiled("combine")(product[i],c[i],float(alpha),float(beta))
                else:product=compiled("combine")(product,c,float(alpha),float(beta))
            with torch._C._DisableTorchDispatch(),torch.no_grad():
                result=torch.from_numpy(product).to(left.dtype)
            self.emit("COMPILED_CPU_END",**call,elapsed_seconds=time.perf_counter()-started,
                      output_shape=list(result.shape),dtype=str(result.dtype))
            return result
        except BaseException as exc:
            self.failed=True;self.emit("COMPILED_CPU_ERROR",**call,error_type=type(exc).__name__)
            raise
