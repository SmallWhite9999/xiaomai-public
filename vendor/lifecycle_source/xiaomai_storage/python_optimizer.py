"""Readable single-group SGD/momentum equations on caller-owned CPU parameters.

No torch.optim step, foreach/fused optimizer, or native profiler wrapper is used.
Numerical updates are prepared with Python scalars and explicit dtype rounding;
public Tensor.copy_ only binds the final values and maintains native versions.
This is a first-order finite FP32/FP64 optimizer with versioned plain-data state,
not a replacement for arbitrary True learners, schedulers or multi-group Adam.
"""
from __future__ import annotations
import hashlib,math
import torch
from . import kernels as k,numeric_words as w

FORMAT='XIAOMAI_PYTHON_SGD_V1'

class OptimizerError(RuntimeError):pass

def values(t):
    k.admit(t);raw=k.raw(t);width=t.element_size()
    return tuple(w.decode(bytes(raw[(p:=k.offset(t,ix)*width):p+width]),t.dtype) for ix in k.coords(t.shape))

def tensor(shape,data,dtype):
    t=k.allocate(shape,dtype=dtype,memory_format=torch.contiguous_format);raw=k.raw(t);width=t.element_size()
    for j,x in enumerate(data):raw[j*width:(j+1)*width]=w.encode_scalar(x,dtype)
    return t

def sha(t):return hashlib.sha256(k.logical_bytes(t)).hexdigest()

def rounded(x,dtype):
    y=w.decode(w.encode_scalar(x,dtype),dtype)
    if not math.isfinite(y):raise OptimizerError('NONFINITE_UPDATE_NOT_COMMITTED')
    return y

class PythonSGD:
    def __init__(self,params,*,lr=.001,momentum=0.,dampening=0.,weight_decay=0.,nesterov=False,maximize=False,sink=None):
        self.params=list(params);self.sink=sink;self.state={};self.steps=0;self.failed=False
        if not self.params or len({id(p) for p in self.params})!=len(self.params):raise OptimizerError('UNIQUE_PARAMETERS_REQUIRED')
        self.options=dict(lr=lr,momentum=momentum,dampening=dampening,weight_decay=weight_decay,nesterov=nesterov,maximize=maximize)
        self.validate_options(self.options)
        storage=set()
        for p in self.params:
            k.admit(p)
            if not p.is_leaf or not p.is_contiguous() or not any(p.dtype is d for d in (torch.float32,torch.float64)):
                raise OptimizerError('CONTIGUOUS_CPU_FLOAT_LEAF_REQUIRED')
            key=p.untyped_storage()._cdata
            if key in storage:raise OptimizerError('SHARED_PARAMETER_STORAGE_NOT_ADMITTED')
            storage.add(key)
    @staticmethod
    def validate_options(o):
        if set(o)!={'lr','momentum','dampening','weight_decay','nesterov','maximize'}:raise OptimizerError('OPTIONS_SCHEMA')
        for key in ('lr','momentum','dampening','weight_decay'):
            v=o[key]
            if not any(type(v) is t for t in (int,float)) or not math.isfinite(v) or v<0:raise OptimizerError('FINITE_NONNEGATIVE_OPTION')
        if type(o['nesterov']) is not bool or type(o['maximize']) is not bool:raise OptimizerError('BOOL_OPTIONS')
        if o['nesterov'] and (o['momentum']<=0 or o['dampening']!=0):raise OptimizerError('NESTEROV_OPTIONS')
    def emit(self,kind,**f):
        if self.sink is not None:self.sink(kind,**f)
    def zero_grad(self,set_to_none=True):
        if type(set_to_none) is not bool:raise OptimizerError('BOOL_ZERO_GRAD_REQUIRED')
        with torch.no_grad():
            for p in self.params:
                if set_to_none:p.grad=None
                elif p.grad is not None:p.grad.zero_()
    def state_dict(self):
        return {'format':FORMAT,'options':dict(self.options),'steps':self.steps,
                'parameters':[{'shape':list(p.shape),'dtype':str(p.dtype)} for p in self.params],
                'momentum':{i:t.detach().clone() for i,t in self.state.items()}}
    def load_state_dict(self,data):
        if type(data) is not dict or set(data)!={'format','options','steps','parameters','momentum'} or data['format']!=FORMAT:
            raise OptimizerError('STATE_FORMAT')
        self.validate_options(data['options'])
        if data['parameters']!=[{'shape':list(p.shape),'dtype':str(p.dtype)} for p in self.params]:raise OptimizerError('PARAMETER_METADATA_MISMATCH')
        if type(data['steps']) is not int or data['steps']<0 or type(data['momentum']) is not dict:raise OptimizerError('STATE_SCHEMA')
        staged={}
        for i,t in data['momentum'].items():
            if type(i) is not int or not 0<=i<len(self.params):raise OptimizerError('MOMENTUM_INDEX')
            p=self.params[i];k.admit(t)
            if t.shape!=p.shape or t.dtype is not p.dtype or not t.is_contiguous() or t.requires_grad:
                raise OptimizerError('MOMENTUM_METADATA_MISMATCH')
            if any(not math.isfinite(v) for v in values(t)):raise OptimizerError('NONFINITE_MOMENTUM')
            staged[i]=t.detach().clone()
        self.options=dict(data['options']);self.steps=data['steps'];self.state=staged;self.failed=False
    def step(self,*,before_commit=None):
        if self.failed:raise OptimizerError('FAILED_OPTIMIZER_REQUIRES_RESTORE')
        self.validate_options(self.options);o=dict(self.options);prepared=[]
        starting_steps=self.steps;momentum_identity={i:(id(t),sha(t)) for i,t in self.state.items()}
        try:
            for i,p in enumerate(self.params):
                g=p.grad
                if g is None:continue
                k.admit(g)
                if g.shape!=p.shape or g.dtype is not p.dtype or g.requires_grad:raise OptimizerError('GRADIENT_METADATA')
                pv,gv=values(p),values(g);buf=values(self.state[i]) if i in self.state else None
                if any(not math.isfinite(x) for x in pv+gv):raise OptimizerError('NONFINITE_INPUT_NOT_REPAIRED')
                update=[];new_buf=[];R=lambda x:rounded(x,p.dtype)
                for j,(param,grad) in enumerate(zip(pv,gv)):
                    d=-grad if o['maximize'] else grad
                    if o['weight_decay']:d=R(d+o['weight_decay']*param)
                    if o['momentum']:
                        b=d if buf is None else R(R(o['momentum']*buf[j])+(1-o['dampening'])*d)
                        new_buf.append(b)
                        direction=R(d+o['momentum']*b) if o['nesterov'] else b
                    else:direction=d
                    after=R(param-o['lr']*direction);update.append(after)
                    self.emit('PY_SGD_ELEMENT',parameter=i,index=j,parameter_before=param.hex(),gradient=grad.hex(),
                        momentum_before=None if buf is None else buf[j].hex(),momentum_after=None if not o['momentum'] else b.hex(),
                        direction=direction.hex(),parameter_after=after.hex())
                prepared.append((i,p,g,sha(p),sha(g),tensor(tuple(p.shape),update,p.dtype),
                                 tensor(tuple(p.shape),new_buf,p.dtype) if o['momentum'] else None))
            if before_commit is not None:before_commit()
            if self.options!=o or self.steps!=starting_steps or momentum_identity!={i:(id(t),sha(t)) for i,t in self.state.items()}:
                raise OptimizerError('OPTIMIZER_STATE_CHANGED_BEFORE_COMMIT')
            for _,p,g,ph,gh,_,_ in prepared:
                if p.grad is not g:raise OptimizerError('GRADIENT_BINDING_CHANGED_BEFORE_COMMIT')
                if sha(p)!=ph or sha(g)!=gh:raise OptimizerError('SOURCE_CHANGED_BEFORE_COMMIT')
            self.emit('PY_SGD_PREPARED',parameters=len(prepared),step=self.steps+1,native_optimizer=False)
            with torch.no_grad():
                for i,p,g,ph,gh,target,buffer in prepared:
                    if sha(p)!=ph or sha(g)!=gh:raise OptimizerError('SOURCE_CHANGED_DURING_COMMIT_REQUIRES_ROLLBACK')
                    p.copy_(target)
                    if buffer is not None:self.state[i]=buffer
            self.steps+=1;self.emit('PY_SGD_COMMITTED',step=self.steps,parameters=len(prepared),native_optimizer=False)
        except BaseException:
            self.failed=True
            self.emit('PY_SGD_FAILED',partial_write_possible=True,restore_checkpoint_required=True)
            raise
