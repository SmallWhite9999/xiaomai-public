"""Python reverse-mode scheduling on the *actual* ATen forward execution.

Forward arithmetic can run under the existing PythonReferenceBackend. Native
PyTorch autograd is disabled during recording, and no torch.backward/grad call
is used by this module. A storage-version SSA graph handles views and copy_.
This is a first-order, CPU, finite-floating diagnostic tape with explicit limits;
unknown active operators, ambiguous aliases and unsupported mutations reject.
Torch Tensor/storage objects are still the boundary representation, not rewritten.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import contextmanager
import hashlib, math, struct, threading, inspect, weakref
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from . import kernels as k, numeric_words as words
from .backend import schema_values
from .frozen_tape import FrozenParameters, FrozenRef, multiply as frozen_multiply

class TapeError(RuntimeError):pass

@dataclass
class Node:
    op:str
    shape:tuple
    data:tuple
    dtype:object
    parents:tuple
    backward:object
    active:bool

@dataclass
class Store:
    dtype:object
    size:int
    node:int
    owner:object
    initialized:set
    external:bool
    generation:int=0

FLOATS=(torch.float32,torch.float64)

def is_tensor(x):return type(x) is torch.Tensor or type(x) is torch.nn.Parameter

def exact_same(a,b):
    if type(a) is float and type(b) is float:return struct.pack('<d',a)==struct.pack('<d',b)
    return type(a) is type(b) and a==b

class PythonTape(TorchDispatchMode):
    def __init__(self,*,sink=None,record_values=False,max_nodes=200000,max_saved_elements=20_000_000,
                 frozen_parameters=None,frozen_matmul='python',frozen_min_elements=65536,
                 frozen_block_bytes=1048576,skip_inactive=None):
        super().__init__()
        from tools.integration_backend import load
        backend=load();self.core=backend.core
        self._inactive_admitted=frozenset().union(*(getattr(backend,n,frozenset()) for n in
            ('PYTHON_OPERATORS','PYTHON_STORAGE_OPERATORS','PYTHON_VIEW_PLAN_OPERATORS',
             'PYTHON_INDEX_OPERATORS','PYTHON_ALLOCATION_OPERATORS','STRUCTURAL')))
        self.sink=sink;self.record_values=record_values;self.max_nodes=max_nodes;self.max_saved_elements=max_saved_elements
        if type(max_nodes) is not int or max_nodes<=0 or type(max_saved_elements) is not int or max_saved_elements<=0:
            raise TapeError('POSITIVE_TAPE_LIMITS_REQUIRED')
        self.nodes=[];self.stores={};self.bindings={};self.retained=[];self.leaves={};self.saved=0
        # Weak storage ownership does not keep discarded inactive intermediates
        # alive. A view shares its storage wrapper; a recycled address cannot
        # inherit ownership after the previous storage has been released.
        self._inactive_owned=weakref.WeakValueDictionary()
        self.failed=False;self.closed=False;self.backward_done=False;self.ops={};self.gradients={}
        self.owner_thread=None;self._no_grad=None
        self._recording=False;self._frozen_depth=0;self.frozen_ops={}
        if frozen_matmul not in ('python','compiled'):raise TapeError('EXPLICIT_FROZEN_MATMUL_MODE_REQUIRED')
        if type(frozen_min_elements) is not int or frozen_min_elements<1 or type(frozen_block_bytes) is not int or frozen_block_bytes<1:
            raise TapeError('POSITIVE_FROZEN_LIMITS_REQUIRED')
        if frozen_parameters is not None and type(frozen_parameters) is not dict:raise TapeError('NAMED_FROZEN_PARAMETER_DICT_REQUIRED')
        self._frozen_parameters=dict(frozen_parameters or {})
        self.frozen_matmul=frozen_matmul;self.frozen_min_elements=frozen_min_elements
        self.frozen_block_bytes=frozen_block_bytes;self.frozen=None;self.inactive_ops={}
        self.skip_inactive=bool(self._frozen_parameters) if skip_inactive is None else skip_inactive
        if type(self.skip_inactive) is not bool:raise TapeError('EXPLICIT_SKIP_INACTIVE_BOOL_REQUIRED')
    def emit(self,kind,**f):
        if self.sink is not None:
            try:self.sink(kind,**f)
            except BaseException:self.failed=True;raise
    def __enter__(self):
        if self._recording or self.closed:raise TapeError('TAPE_SINGLE_FORWARD_SCOPE_REQUIRED')
        if torch.is_inference_mode_enabled():raise TapeError('INFERENCE_MODE_NOT_ADMITTED')
        self.owner_thread=threading.get_ident()
        self._no_grad=torch.no_grad();self._no_grad.__enter__()
        try:self.emit('PY_TAPE_POLICY',native_autograd=False,order=1,storage_ssa=True,
            max_nodes=self.max_nodes,max_saved_elements=self.max_saved_elements,
            dtype_domain=['float32','float64'],record_values=self.record_values,native_boundary='Tensor/storage objects; not backward engine')
        except BaseException:
            self._no_grad.__exit__(*__import__('sys').exc_info());raise
        try:
            if self._frozen_parameters:
                self.frozen=FrozenParameters(self._frozen_parameters,min_elements=self.frozen_min_elements,
                    block_bytes=self.frozen_block_bytes,emit=self.emit,reject_hooks=self.reject_hooks)
            result=super().__enter__();self._recording=True;return result
        except BaseException:
            self._no_grad.__exit__(*__import__('sys').exc_info());raise
    def __exit__(self,*args):
        try:
            if args[0] is None and self.frozen:
                try:self.frozen.verify_all()
                except BaseException:self.failed=True;raise
        finally:
            try:super().__exit__(*args)
            finally:
                self._recording=False
                self._no_grad.__exit__(*args);self.closed=True;self.failed|=args[0] is not None
                self.emit('PY_TAPE_FORWARD_CLOSED',failed=self.failed,nodes=len(self.nodes),saved_elements=self.saved,
                          operators=self.ops,inactive_passthrough=self.inactive_ops)
    @contextmanager
    def frozen_region(self, *, reason):
        """Explicit no-derivative scope on this tape, retaining lower backends.

        Intended for the pinned research runtime's already-frozen backbone
        feature extraction and inference state. It never enables native grad,
        bypasses lower numerical dispatch, or changes any parameter flag.
        Return detached/ordinary outputs; externally returned Parameter objects
        retain their own identity and are not magically converted to constants.
        """
        if (not self._recording or self.closed or self.failed
                or threading.get_ident()!=self.owner_thread):
            raise TapeError('ACTIVE_OWNER_TAPE_REQUIRED_FOR_FROZEN_REGION')
        if type(reason) is not str or not reason:raise TapeError('EXPLICIT_FROZEN_REASON_REQUIRED')
        self.emit('PY_TAPE_FROZEN_BEGIN',reason=reason,depth=self._frozen_depth+1,
                  native_backward=False,numerical_dispatch_bypassed=False)
        self._frozen_depth+=1
        try:yield
        except BaseException:
            self.failed=True;raise
        finally:
            self._frozen_depth-=1
            self.emit('PY_TAPE_FROZEN_END',reason=reason,depth=self._frozen_depth)
    def node(self,op,shape,data,dtype,parents=(),backward=None,active=None,observed=None):
        shape=tuple(shape)
        if len(self.nodes)>=self.max_nodes or self.saved+math.prod(shape)>self.max_saved_elements:raise TapeError('TAPE_RESOURCE_LIMIT')
        data=tuple(data)
        if len(data)!=math.prod(shape):raise TapeError('NODE_SHAPE')
        if len(self.nodes)>=self.max_nodes or self.saved+len(data)>self.max_saved_elements:raise TapeError('TAPE_RESOURCE_LIMIT')
        if active is None:active=any(self.nodes[p].active for p in parents)
        n=len(self.nodes);self.nodes.append(Node(op,shape,data,dtype,tuple(parents),backward,active));self.saved+=len(data)
        fields={}
        if self.record_values:fields['values']=[self.encode_number(x) if observed is None or j in observed else {'unread_storage_slot':True} for j,x in enumerate(data)]
        self.emit('PY_TAPE_NODE',node=n,operation=op,parents=list(parents),shape=list(shape),dtype=str(dtype),active=active,**fields)
        return n
    @staticmethod
    def encode_number(x):
        if type(x) is float:return {'f64_le':struct.pack('<d',x).hex()}
        if type(x) is bool:return {'bool':x}
        if type(x) is int:return {'int':str(x)}
        raise TapeError('BUILTIN_NUMBER_REQUIRED')
    def values(self,t):
        k.admit(t);buf=k.raw(t);width=t.element_size()
        return tuple(words.decode(bytes(buf[(p:=k.offset(t,ix)*width):p+width]),t.dtype) for ix in k.coords(t.shape))
    def positions(self,t):return tuple(k.offset(t,ix) for ix in k.coords(t.shape))
    def key(self,t):return int(t.untyped_storage()._cdata)
    @staticmethod
    def reject_hooks(t):
        # Only exact admitted Tensor/Parameter metadata is inspected. Native
        # backward hooks cannot silently disappear in the Python backward path.
        # Custom Function behavior and arbitrary external graph hooks remain
        # outside this tape's first-order, replayed-operation contract.
        for name in ('_backward_hooks','_post_accumulate_grad_hooks'):
            hooks=getattr(t,name)
            if hooks is None:continue
            if type(hooks) is dict:
                present=dict.__len__(hooks)!=0
            elif type(hooks) is OrderedDict:
                present=OrderedDict.__len__(hooks)!=0
            else:
                raise TapeError('TENSOR_BACKWARD_HOOK_METADATA_NOT_ADMITTED')
            if present:raise TapeError('TENSOR_BACKWARD_HOOKS_NOT_ADMITTED')
    def register(self,t,*,external=False):
        k.admit(t)
        self.reject_hooks(t)
        if id(t) not in self.bindings and external and t.requires_grad and not t.is_leaf:
            raise TapeError('EXTERNAL_NONLEAF_REQUIRES_EXPLICIT_REPLAY')
        key=self.key(t);self.retained.append(t)
        if key in self.stores and id(t) not in self.bindings and external and t.requires_grad:
            raise TapeError('INDEPENDENT_ACTIVE_LEAVES_SHARE_STORAGE')
        if key not in self.stores:
            size=t.untyped_storage().nbytes()//t.element_size()
            if t.untyped_storage().nbytes()%t.element_size():raise TapeError('MISALIGNED_STORAGE')
            if self.saved+size>self.max_saved_elements:raise TapeError('TAPE_RESOURCE_LIMIT')
            positions=self.positions(t);values=self.values(t);data=[0]*size
            for p,v in zip(positions,values):data[p]=v
            requires=bool(t.requires_grad) and external
            if requires and not any(t.dtype is d for d in FLOATS):raise TapeError('ACTIVE_DTYPE_NOT_SUPPORTED')
            n=self.node('external_storage' if external else 'constant_storage',(size,),data,t.dtype,active=requires,observed=set(positions))
            self.stores[key]=Store(t.dtype,size,n,t,set(positions),external)
            if requires:self.leaves[id(t)]=(t,n,positions)
        store=self.stores[key]
        if store.dtype is not t.dtype:raise TapeError('MIXED_DTYPE_STORAGE_ALIAS_NOT_SUPPORTED')
        self.bindings.setdefault(id(t),(key,False))
        return store
    def read(self,t):
        if self.frozen:
            ref=self.frozen.ref(t)
            if ref is not None:return ref
        store=self.register(t,external=self.key(t) not in self._inactive_owned)
        key,stop=self.bindings[id(t)];positions=self.positions(t);values=self.values(t)
        current=self.nodes[store.node]
        # Reject edits not recorded as a supported operation; compare consumed
        # values, not just version counters which can be bypassed with raw bytes.
        for p,v in zip(positions,values):
            if p in store.initialized and not exact_same(current.data[p],v):raise TapeError('UNRECORDED_STORAGE_CHANGE')
            if p not in store.initialized:
                if not store.external or current.active:raise TapeError('UNINITIALIZED_OR_UNTRACKED_ALIAS_READ')
                data=list(current.data);data[p]=v;current.data=tuple(data);store.initialized.add(p)
        if stop:return self.node('stop_gradient',t.shape,values,t.dtype,active=False)
        parent=store.node
        def vjp(g):
            out=[0.0]*store.size
            for p,v in zip(positions,g):out[p]+=v
            return (tuple(out),)
        return self.node('storage_gather',t.shape,values,t.dtype,(parent,),vjp)
    def args_tree(self,value):
        if is_tensor(value):return self.read(value)
        if any(type(value) is t for t in (list,tuple)):return [self.args_tree(x) if is_tensor(x) or any(type(x) is t for t in (list,tuple)) else None for x in value]
        return None
    def bind_result(self,t,logical_node):
        k.admit(t);key=self.key(t);positions=self.positions(t);values=self.values(t)
        if key in self.stores:raise TapeError('UNDECLARED_OUTPUT_ALIAS')
        size=t.untyped_storage().nbytes()//t.element_size()
        if self.saved+size>self.max_saved_elements:raise TapeError('TAPE_RESOURCE_LIMIT')
        data=[0]*size
        if len(set(positions))!=len(positions):raise TapeError('OVERLAPPING_NEW_RESULT')
        for p,v in zip(positions,values):data[p]=v
        def vjp(g):return (tuple(g[p] for p in positions),)
        n=self.node('storage_result',(size,),data,t.dtype,(logical_node,),vjp,observed=set(positions))
        self.stores[key]=Store(t.dtype,size,n,t,set(positions),False)
        self.retained.append(t);self.bindings[id(t)]=(key,False)
    def bind_view(self,t,*,stop=False):
        if self.frozen:
            ref=self.frozen.ref(t)
            if ref is not None:
                self.emit('PY_FROZEN_VIEW',shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),
                          parameter_names=ref.owner.names,storage_sha256=ref.owner.sha256)
                return
        key=self.key(t)
        if key not in self.stores:raise TapeError('VIEW_CHANGED_STORAGE')
        if self.stores[key].dtype is not t.dtype:raise TapeError('VIEW_DTYPE_REINTERPRET_NOT_ADMITTED')
        self.retained.append(t);self.bindings[id(t)]=(key,stop)
    def mutate_copy(self,target,src,src_node):
        key=self.key(target);store=self.stores[key]
        if self.bindings[id(target)][1] and self.nodes[store.node].active:raise TapeError('DETACHED_ACTIVE_ALIAS_MUTATION_NOT_ADMITTED')
        positions=self.positions(target)
        if len(set(positions))!=len(positions):raise TapeError('OVERLAPPING_MUTATION')
        source=self.nodes[src_node];outshape=tuple(target.shape)
        mapping=self.broadcast_map(source.shape,outshape)
        before=store.node;data=list(self.nodes[before].data)
        after_values=self.values(target)
        for p,v in zip(positions,after_values):data[p]=v
        def vjp(g):
            old=list(g);sg=[0.0]*len(source.data)
            for p,j in zip(positions,mapping):old[p]=0.0;sg[j]+=g[p]
            return tuple(old),tuple(sg)
        store.node=self.node('copy_storage_update',(store.size,),data,store.dtype,(before,src_node),vjp,observed=store.initialized|set(positions))
        store.initialized.update(positions);store.generation+=1
    def broadcast_map(self,shape,outshape):
        if self.core.broadcast_shape(shape,outshape)!=outshape:raise TapeError('BROADCAST_SHAPE')
        result=[]
        for ix in k.coords(outshape):
            sub=ix[len(ix)-len(shape):] if shape else ();p=0
            for j,n in zip(sub,shape):p=p*n+(0 if n==1 else j)
            result.append(p)
        return tuple(result)
    def unbroadcast(self,grad,outshape,inshape):
        target=[0.0]*math.prod(inshape)
        for p,g in zip(self.broadcast_map(inshape,outshape),grad):target[p]+=g
        return tuple(target)
    def array(self,n):
        n=self.nodes[n];return self.core.Array(n.shape,n.data)
    def metadata(self,value):
        if value is None or any(type(value) is t for t in (str,int,bool)):return value
        if type(value) is float:return self.encode_number(value)
        if any(type(value) is t for t in (list,tuple)):return [self.metadata(v) for v in value]
        if is_tensor(value):return {'tensor_binding':id(value)}
        return {'opaque_type_id':id(type(value))}
    def node_ids(self,value):
        if type(value) is FrozenRef:
            return {'frozen_parameter_names':value.owner.names,'shape':list(value.shape),
                    'dtype':str(value.dtype),'storage_sha256':value.owner.sha256}
        if type(value) is dict:return {n:self.node_ids(v) for n,v in value.items()}
        if type(value) in (list,tuple):return [self.node_ids(v) for v in value]
        return value
    def _op(self,name,v,ids,result):
        # Capture only arguments actually Tensor-valued. Scalar metadata remains
        # inert Python values; no arbitrary user equality/property evaluation.
        parents=[];names=[]
        for key,ident in ids.items():
            if type(ident) is int and self.nodes[ident].active:parents.append(ident);names.append(key)
            elif type(ident) is list:
                for j,n in enumerate(ident):
                    if type(n) is int and self.nodes[n].active:parents.append(n);names.append((key,j))
        active=any(self.nodes[n].active for n in parents)
        if name in self.NOGRAD:active=False
        if not is_tensor(result):
            if active:raise TapeError('ACTIVE_SCALAR_CONTROL_FLOW_NOT_SUPPORTED:'+name)
            return
        if active and not any(result.dtype is d for d in FLOATS):raise TapeError('DISCRETE_ACTIVE_OUTPUT_NOT_SUPPORTED:'+name)
        values=self.values(result)
        output_shape=tuple(result.shape)
        def back(g):
            res=self.vjp(name,v,ids,output_shape,g)
            return tuple(res[key] if key in res else tuple(0.0 for _ in self.nodes[n].data) for key,n in zip(names,parents))
        if active:self.check_vjp(name)
        metadata={key:self.metadata(value) for key,value in v.items() if not is_tensor(value)}
        self.emit('PY_OPERATION_BIND',operator=name,input_nodes=self.node_ids(ids),metadata=metadata,output_shape=list(output_shape))
        n=self.node(name,result.shape,values,result.dtype,parents,back,active)
        self.bind_result(result,n)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if not self.frozen:return self._dispatch(func,types,args,kwargs)
        values=schema_values(func,args,kwargs or {})
        tensors=tuple(self._tensors(tuple(values.values())))
        touched=self.frozen.touched(tensors)
        try:
            if func._schema.is_mutable:
                writes=[values[a.name] for a in func._schema.arguments
                        if a.alias_info is not None and a.alias_info.is_write]
                if self.frozen.touched(self._tensors(writes)):
                    raise TapeError('FROZEN_PARAMETER_MUTATION_NOT_ADMITTED')
            is_view=str(func) in self.VIEWS
            for store in touched:
                store.verify_metadata() if is_view else store.verify()
            result=self._dispatch(func,types,args,kwargs)
            for store in touched:
                store.verify_metadata() if is_view else store.verify()
            return result
        except BaseException:
            self.failed=True;raise
    def _dispatch(self,func,types,args=(),kwargs=None):
        if threading.get_ident()!=self.owner_thread:raise TapeError('TAPE_THREAD_MISMATCH')
        if torch.is_grad_enabled():raise TapeError('NATIVE_AUTOGRAD_REENABLED_INSIDE_PYTHON_TAPE')
        if self._frozen_depth:
            name=str(func);self.frozen_ops[name]=self.frozen_ops.get(name,0)+1
            # No-derivative does not mean that aliases may be forgotten. A
            # detached view of an already active storage must remain a stopped
            # edge when consumed after this scope, without stopping its source.
            # Untracked feature matrices are not materialized on the tape.
            with torch._C._DisableTorchDispatch():
                if func._schema.is_mutable:
                    values=schema_values(func,args,kwargs or {})
                    writes=[values[a.name] for a in func._schema.arguments
                            if a.alias_info is not None and a.alias_info.is_write]
                    if any(self.key(t) in self.stores for t in self._tensors(writes)):
                        raise TapeError('FROZEN_MUTATION_OF_TRACKED_STORAGE_NOT_ADMITTED')
            result=func(*args,**(kwargs or {}))
            with torch._C._DisableTorchDispatch():
                for t in self._tensors(result):
                    if self.key(t) in self.stores:
                        if id(t) in self.bindings and not self.bindings[id(t)][1]:
                            raise TapeError('FROZEN_ALIAS_REUSES_TRACKED_OBJECT')
                        self.bind_view(t,stop=True)
            return result
        kwargs=kwargs or {};name=str(func);self.ops[name]=self.ops.get(name,0)+1
        v=schema_values(func,args,kwargs)
        try:
            tensor_args=tuple(self._tensors(tuple(v.values())))
            # requires_grad is insufficient: active intermediate Tensors have
            # it cleared by native no_grad. Any already tracked storage follows
            # the ordinary SSA path even if its current edge is stopped.
            inactive=(self.skip_inactive and name in self._inactive_admitted
                    and not any(self.key(t) in self.stores or t.requires_grad for t in tensor_args))
            if inactive and func._schema.is_mutable:
                # Only scratch newly allocated inside this tape is writable
                # without an SSA record. External tensors, source aliases and
                # anything already consumed by the graph stay on guarded paths.
                writes=tuple(self._tensors([v[a.name] for a in func._schema.arguments
                    if a.alias_info is not None and a.alias_info.is_write]))
                inactive=(name in ('aten.copy_.default','aten.masked_fill_.Scalar') and bool(writes)
                    and all(self.key(t) in self._inactive_owned and k.nonoverlap(t) for t in writes))
                if inactive and name=='aten.copy_.default':
                    if not is_tensor(v['src']) or v.get('non_blocking'):raise TapeError('COPY_DOMAIN')
                    if self.key(v['self'])==self.key(v['src']):
                        dp=self.positions(v['self']);sp=self.positions(v['src'])
                        if set(dp)&set(sp) and dp!=sp:raise TapeError('PARTIAL_OVERLAPPING_COPY_NOT_SUPPORTED')
                if inactive and name=='aten.masked_fill_.Scalar' and type(v['value']) not in (float,int,bool):
                    raise TapeError('BUILTIN_MASKED_FILL_REQUIRED')
            if inactive:
                input_keys={self.key(t) for t in tensor_args}
                result=func(*args,**kwargs)
                for t in self._tensors(result):
                    key=self.key(t)
                    if key not in input_keys:self._inactive_owned[key]=t.untyped_storage()
                self.inactive_ops[name]=self.inactive_ops.get(name,0)+1
                self.emit('PY_INACTIVE_FORWARD',operator=name,derivative_recorded=False,
                          lower_numerical_dispatch_bypassed=False)
                return result
            # capture logical input versions before any mutation
            with torch._C._DisableTorchDispatch():ids={key:self.args_tree(value) for key,value in v.items()}
            active=any(self.nodes[n].active for n in self._flatten_ids(ids.values()))
            if active and name=='aten.gather.default' and v.get('sparse_grad',False):
                raise TapeError('SPARSE_GATHER_GRADIENT_NOT_ADMITTED')
            if (active and name not in ('aten.mm.default','aten.bmm.default','aten.addmm.default','aten.baddbmm.default')
                    and any(type(ident) is FrozenRef and math.prod(ident.shape)>self.frozen_min_elements for ident in ids.values())):
                raise TapeError('LAZY_FROZEN_ACTIVE_OPERATOR_NOT_ADMITTED:'+name)
            mutable=func._schema.is_mutable
            if mutable and name not in ('aten.copy_.default','aten.masked_fill_.Scalar'):raise TapeError('MUTATION_NOT_SUPPORTED:'+name)
            if name=='aten.masked_fill_.Scalar':
                target=v['self'];destination=self.stores[self.key(target)]
                if destination.external or not k.nonoverlap(target):raise TapeError('MASKED_FILL_EXTERNAL_OR_OVERLAPPING_MUTATION')
                if self.bindings[id(target)][1] and self.nodes[destination.node].active:raise TapeError('DETACHED_ACTIVE_ALIAS_MUTATION_NOT_ADMITTED')
                if type(v['value']) not in (float,int,bool):raise TapeError('BUILTIN_MASKED_FILL_REQUIRED')
                if len(self.nodes)+1>self.max_nodes or self.saved+destination.size>self.max_saved_elements:raise TapeError('TAPE_RESOURCE_LIMIT_BEFORE_MUTATION')
            if name=='aten.copy_.default':
                if not is_tensor(v['src']) or v.get('non_blocking'):raise TapeError('COPY_DOMAIN')
                if not k.nonoverlap(v['self']):raise TapeError('OVERLAPPING_MUTATION')
                destination=self.stores[self.key(v['self'])]
                if destination.external:raise TapeError('EXTERNAL_SOURCE_MUTATION_NOT_ADMITTED')
                if self.bindings[id(v['self'])][1] and self.nodes[destination.node].active:
                    raise TapeError('DETACHED_ACTIVE_ALIAS_MUTATION_NOT_ADMITTED')
                if len(self.nodes)+1>self.max_nodes or self.saved+destination.size>self.max_saved_elements:
                    raise TapeError('TAPE_RESOURCE_LIMIT_BEFORE_MUTATION')
                if self.key(v['self'])==self.key(v['src']):
                    dp=self.positions(v['self']);sp=self.positions(v['src'])
                    if set(dp)&set(sp) and dp!=sp:raise TapeError('PARTIAL_OVERLAPPING_COPY_NOT_SUPPORTED')
            frame=inspect.currentframe().f_back
            location={'file':frame.f_code.co_filename,'line':frame.f_lineno,'function':frame.f_code.co_name};del frame
            self.emit('PY_FORWARD_CALL',operator=name,source=location,active_inputs=active)
            result=func(*args,**kwargs)
            with torch._C._DisableTorchDispatch():
                if mutable:
                    if name=='aten.masked_fill_.Scalar':self.mutate_masked_fill(v['self'],ids['mask'])
                    else:self.mutate_copy(v['self'],v['src'],ids['src'])
                    return result
                tensors=list(self._tensors(result))
                view=name in self.VIEWS
                if view:
                    # detach cuts this logical edge but leaves its physical alias.
                    inherited=bool(is_tensor(v.get('self')) and id(v['self']) in self.bindings and self.bindings[id(v['self'])][1])
                    for t in tensors:
                        cut=(name=='aten.detach.default' and self.key(t) in self.stores and self.nodes[self.stores[self.key(t)].node].active)
                        self.bind_view(t,stop=cut or inherited)
                else:
                    if any(type(result) is t for t in (tuple,list)) and name=='aten.native_layer_norm.default':
                        self.layernorm_result(v,ids,result)
                    elif any(type(result) is t for t in (tuple,list)) and name=='aten.nll_loss_forward.default':
                        self.nll_result(v,ids,result)
                    elif any(type(result) is t for t in (tuple,list)):
                        if active:raise TapeError('ACTIVE_MULTI_OUTPUT_NOT_SUPPORTED:'+name)
                        for t in tensors:self.register(t,external=False)
                    elif is_tensor(result):self._op(name,v,ids,result)
                    elif active and name not in self.NOGRAD:raise TapeError('ACTIVE_NON_TENSOR_RESULT:'+name)
            return result
        except BaseException as exc:
            self.failed=True;self.emit('PY_TAPE_ERROR',operator=name,error_type=type(exc).__name__);raise
    def layernorm_result(self,v,ids,result):
        C=self.core;A=C.Array
        keys=[name for name in ('input','weight','bias') if type(ids.get(name)) is int]
        parents=tuple(ids[name] for name in keys)
        X=self.array(ids['input']);W=self.array(ids['weight']) if type(ids.get('weight')) is int else None
        B=self.array(ids['bias']) if type(ids.get('bias')) is int else None
        M=A(tuple(result[1].shape),self.values(result[1]));R=A(tuple(result[2].shape),self.values(result[2]))
        shape=tuple(v['normalized_shape']);output_shape=tuple(result[0].shape)
        def back(g):
            gx,gw,gb=C.layernorm_backward(A(output_shape,tuple(g)),X,shape,M,R,W,B,(True,W is not None,B is not None))
            gradients={'input':gx,'weight':gw,'bias':gb}
            return tuple(gradients[key].data for key in keys)
        n=self.node('aten.native_layer_norm.default',output_shape,self.values(result[0]),result[0].dtype,parents,back)
        self.bind_result(result[0],n)
        # Native statistics outputs are non-differentiable; they are not silently
        # promoted to arbitrary differentiable auxiliary values.
        for t in result[1:]:self.register(t,external=False)
    def mutate_masked_fill(self,target,mask_node):
        store=self.stores[self.key(target)];before=store.node
        mask=self.nodes[mask_node];mapping=self.broadcast_map(mask.shape,tuple(target.shape))
        positions=self.positions(target);logical=self.values(target)
        changed={p for p,j in zip(positions,mapping) if mask.data[j]}
        data=list(self.nodes[before].data)
        for p,v in zip(positions,logical):data[p]=v
        def back(g):return (tuple(0. if i in changed else v for i,v in enumerate(g)),)
        store.node=self.node('masked_fill_storage_update',(store.size,),data,store.dtype,(before,),back,
                             observed=store.initialized|set(positions))
        store.initialized.update(positions);store.generation+=1
        self.emit('PY_MASKED_FILL_MUTATION',before=before,after=store.node,mask_node=mask_node,
                  replaced_storage_positions=sorted(changed),constant_fill_gradient=False)
    def nll_result(self,v,ids,result):
        # NLL's total_weight output is a non-differentiable normalization value.
        # Preserve the actual forward weight; never silently attach a gradient to it.
        parent=ids['self']; X=self.nodes[parent]
        if len(X.shape) not in (1,2):raise TapeError('NLL_RANK_1_2_ONLY')
        target=self.nodes[ids['target']]
        if target.active:raise TapeError('NLL_DISCRETE_TARGET_ACTIVE')
        W=self.nodes[ids['weight']] if type(ids.get('weight')) is int else None
        if W is not None and W.active:raise TapeError('NLL_WEIGHT_GRAD_NOT_ADMITTED')
        classes=X.shape[-1];batch=1 if len(X.shape)==1 else X.shape[0]
        if len(target.data)!=batch:raise TapeError('NLL_TARGET_SHAPE')
        ignore=v['ignore_index'];reduction=v['reduction']
        if reduction not in (0,1,2):raise TapeError('NLL_REDUCTION')
        total=self.values(result[1])[0]
        targets=tuple(target.data)
        if any(type(t) is not int or (t!=ignore and not 0<=t<classes) for t in targets):raise TapeError('NLL_TARGET_RANGE')
        output_shape=tuple(result[0].shape)
        def back(g):
            output=[0.0]*len(X.data)
            for row,t in enumerate(targets):
                if t==ignore:continue
                scale=g[row] if reduction==0 and len(g)>1 else g[0]
                if reduction==1:
                    if total==0 or not math.isfinite(total):raise TapeError('NLL_NONFINITE_NORMALIZATION')
                    scale/=total
                weight=1.0 if W is None else W.data[t]
                output[row*classes+t]=-weight*scale
            return (tuple(output),)
        self.emit('PY_NLL_BIND',input_node=parent,target_node=ids['target'],weight_node=ids.get('weight'),
            reduction=reduction,ignore_index=ignore,total_weight=self.encode_number(total),
            non_differentiable_auxiliary_output=True)
        n=self.node('aten.nll_loss_forward.default',output_shape,self.values(result[0]),result[0].dtype,(parent,),back)
        self.bind_result(result[0],n);self.register(result[1],external=False)
    def _flatten_ids(self,items):
        for x in items:
            if type(x) is int:yield x
            elif type(x) is list:yield from self._flatten_ids(x)
    def _tensors(self,value):
        if is_tensor(value):yield value
        elif any(type(value) is t for t in (list,tuple)):
            for x in value:yield from self._tensors(x)
    VIEWS=frozenset(('aten.view.default','aten._unsafe_view.default','aten.reshape.default',
        'aten.lift.default','aten.lift_fresh.default','aten.expand.default','aten.select.int','aten.slice.Tensor','aten.unsqueeze.default',
        'aten.squeeze.default','aten.squeeze.dim','aten.squeeze.dims','aten.transpose.int',
        'aten.permute.default','aten.t.default','aten.alias.default','aten.detach.default',
        'aten.split.Tensor','aten.split_with_sizes.default','aten.unbind.int','aten.as_strided.default'))
    NOGRAD=frozenset(('aten.zeros_like.default','aten.ones_like.default','aten.empty_like.default',
        'aten.full_like.default','aten.new_zeros.default','aten.new_full.default','aten.new_empty.default',
        # Explicit boolean predicates have no derivative in the native API.
        # Recording their value/branch does not create an active gradient edge.
        'aten.isfinite.default','aten.isinf.default','aten.isnan.default',
        'aten.eq.Tensor','aten.eq.Scalar','aten.ne.Tensor','aten.ne.Scalar',
        'aten.lt.Tensor','aten.lt.Scalar','aten.le.Tensor','aten.le.Scalar',
        'aten.gt.Tensor','aten.gt.Scalar','aten.ge.Tensor','aten.ge.Scalar',
        'aten.all.default','aten.all.dim','aten.all.dims',
        'aten.any.default','aten.any.dim','aten.any.dims',
        'aten.logical_not.default','aten.logical_and.default','aten.logical_or.default','aten.allclose.default'))
    DIFF=frozenset(('aten.add.Tensor','aten.add.Scalar','aten.sub.Tensor','aten.sub.Scalar','aten.rsub.Scalar',
        'aten.mul.Tensor','aten.mul.Scalar','aten.div.Tensor','aten.div.Scalar',
        'aten.neg.default','aten.exp.default','aten.log.default','aten.sin.default','aten.cos.default',
        'aten.tanh.default','aten.sigmoid.default','aten.silu.default','aten.softplus.default',
        'aten.rsqrt.default','aten.sqrt.default','aten.pow.Tensor_Scalar','aten.square.default',
        'aten.mean.dim','aten.mean.default','aten.sum.dim_IntList','aten.sum.default',
        'aten.mm.default','aten.bmm.default','aten.addmm.default','aten.baddbmm.default','aten.gelu.default','aten.embedding.default','aten.convolution.default',
        'aten.cat.default','aten.stack.default','aten.clone.default','aten._to_copy.default','aten.lift_fresh_copy.default',
        'aten.cumsum.default','aten.constant_pad_nd.default','aten.triu.default','aten.tril.default',
        'aten.masked_fill.Scalar','aten._softmax.default','aten._log_softmax.default',
        'aten.mse_loss.default','aten.smooth_l1_loss.default','aten.abs.default','aten.gather.default'))
    def check_vjp(self,name):
        if name not in self.DIFF:raise TapeError('ACTIVE_VJP_NOT_IMPLEMENTED:'+name)
    def vjp(self,name,v,ids,outshape,g):
        C=self.core;A=C.Array;ans={}
        def arr(key):
            ident=ids.get(key)
            if type(ident) is FrozenRef:return ident.materialize(C,self.frozen_min_elements)
            return self.array(ident) if type(ident) is int else C.scalar(v[key])
        def operand(key):
            return ids[key] if type(ids.get(key)) is FrozenRef else arr(key)
        def need(key):return type(ids.get(key)) is int and self.nodes[ids[key]].active
        def put(key,values,shape=outshape):
            if need(key):ans[key]=self.unbroadcast(values,shape,self.nodes[ids[key]].shape)
        def ew(fn,*a):return C.elementwise(fn,*a)
        base=name.split('.')[1]
        G=A(tuple(outshape),tuple(g));x=operand('self') if base in ('mm','bmm','addmm','baddbmm') else arr('self') if 'self' in v else None
        if base in ('add','sub','rsub','mul','div'):
            y=arr('other');alpha=v.get('alpha',1)
            if base=='add':put('self',g);put('other',[alpha*z for z in g])
            elif base=='sub':put('self',g);put('other',[-alpha*z for z in g])
            elif base=='rsub':put('self',[-alpha*z for z in g]);put('other',g)
            elif base=='mul':put('self',ew(lambda a,b:a*b,G,y).data);put('other',ew(lambda a,b:a*b,G,x).data)
            else:
                if v.get('rounding_mode') is not None:raise TapeError('ROUNDED_DIVISION_VJP_NOT_SUPPORTED')
                put('self',ew(lambda a,b:a/b,G,y).data);put('other',ew(lambda a,b,c:-a*b/(c*c),G,x,y).data)
        elif base in ('mse_loss','smooth_l1_loss'):
            target=arr('target');shape=C.broadcast_shape(x.shape,target.shape)
            delta=ew(lambda a,b:a-b,x,target)
            reduction=v['reduction'];factor=math.prod(shape) if reduction==1 else 1
            if reduction not in (0,1,2) or factor==0:raise TapeError('LOSS_REDUCTION_OR_EMPTY_MEAN')
            beta=v.get('beta',1.0)
            vals=[]
            for j,difference in enumerate(delta.data):
                upstream=g[j] if reduction==0 else g[0]
                if base=='mse_loss':local=2*difference
                elif not math.isfinite(difference):local=float('nan')
                elif beta==0:local=1. if difference>0 else -1. if difference<0 else 0.
                else:local=difference/beta if abs(difference)<beta else (1. if difference>0 else -1.)
                vals.append(upstream*local/factor)
            ans['self']=self.unbroadcast(vals,shape,x.shape)
            ans['target']=self.unbroadcast(tuple(-v for v in vals),shape,target.shape)
        elif base=='abs':
            put('self',tuple(up*(1. if z>0 else -1. if z<0 else 0.) for up,z in zip(g,x.data)))
        elif base in ('clone','_to_copy','lift_fresh_copy'):put('self',g)
        elif base in ('neg','exp','log','sin','cos','tanh','sigmoid','silu','softplus','rsqrt','sqrt','pow','square','gelu'):
            def derivative(z):
                if base=='neg':return -1.
                if base=='exp':return C.exp(z)
                if base=='log':return C.divide(1,z)
                if base=='sin':return math.cos(z)
                if base=='cos':return -math.sin(z)
                if base=='tanh':return 1-math.tanh(z)**2
                if base=='sigmoid':s=C.sigmoid(z);return s*(1-s)
                if base=='silu':s=C.sigmoid(z);return s+z*s*(1-s)
                if base=='softplus':return 1. if v['beta']*z>v['threshold'] else C.sigmoid(v['beta']*z)
                if base=='rsqrt':return -.5*C.rsqrt(z)**3
                if base=='sqrt':return C.divide(.5,C.sqrt(z))
                if base=='gelu':
                    if v['approximate']=='none':return .5*(1+math.erf(z/math.sqrt(2)))+z*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
                    if v['approximate']=='tanh':
                        c=math.sqrt(2/math.pi);t=math.tanh(c*(z+.044715*z**3))
                        return .5*(1+t)+.5*z*(1-t*t)*c*(1+3*.044715*z*z)
                    raise TapeError('GELU_APPROXIMATION')
                exponent=2 if base=='square' else v['exponent']
                return 0. if exponent==0 else exponent*C.power(z,exponent-1)
            put('self',ew(lambda a,b:a*derivative(b),G,x).data)
        elif base in ('sum','mean'):
            shape=x.shape;dims=v.get('dim')
            if dims is None or len(dims)==0:dims=tuple(range(len(shape)))
            dims=tuple(d%len(shape) for d in dims);keep=v.get('keepdim',False)
            expanded=G if keep else G.reshape(tuple(1 if d in dims else shape[d] for d in range(len(shape))))
            factor=math.prod(shape[d] for d in dims) if base=='mean' else 1
            vals=tuple(C.bat(expanded,ix)/factor for ix in k.coords(shape))
            ans['self']=vals
        elif base in ('mm','bmm'):
            y=operand('mat2')
            def tr(a):
                shape=a.shape[:-2]+(a.shape[-1],a.shape[-2])
                return A(shape,tuple(a.at(ix[:-2]+(ix[-1],ix[-2])) for ix in k.coords(shape)))
            fn=C.mm if base=='mm' else C.bmm
            if need('self'):
                ans['self']=(frozen_multiply(C,G,y,transpose_right=True,batch=base=='bmm',mode=self.frozen_matmul,
                    emit=self.emit,record_values=self.record_values) if type(y) is FrozenRef or self.frozen_matmul=='compiled' else fn(G,tr(y))).data
            if need('mat2'):
                ans['mat2']=(frozen_multiply(C,x,G,transpose_left=True,batch=base=='bmm',mode=self.frozen_matmul,
                    emit=self.emit,record_values=self.record_values) if type(x) is FrozenRef or self.frozen_matmul=='compiled' else fn(tr(x),G)).data
        elif base in ('addmm','baddbmm'):
            batch=base=='baddbmm';ka='batch1' if batch else 'mat1';kb='batch2' if batch else 'mat2'
            a,b=operand(ka),operand(kb);alpha=v['alpha'];beta=v['beta']
            def tr(z):
                sh=z.shape[:-2]+(z.shape[-1],z.shape[-2])
                return A(sh,tuple(z.at(ix[:-2]+(ix[-1],ix[-2])) for ix in k.coords(sh)))
            fn=C.bmm if batch else C.mm
            if need(ka):
                product=frozen_multiply(C,G,b,transpose_right=True,batch=batch,mode=self.frozen_matmul,
                    emit=self.emit,record_values=self.record_values) if type(b) is FrozenRef or self.frozen_matmul=='compiled' else fn(G,tr(b))
                ans[ka]=tuple(alpha*z for z in product.data)
            if need(kb):
                product=frozen_multiply(C,a,G,transpose_left=True,batch=batch,mode=self.frozen_matmul,
                    emit=self.emit,record_values=self.record_values) if type(a) is FrozenRef or self.frozen_matmul=='compiled' else fn(tr(a),G)
                ans[kb]=tuple(alpha*z for z in product.data)
            put('self',tuple(beta*z for z in g))
        elif base=='embedding':
            weight=arr('weight');idx=arr('indices')
            ans['weight']=C.embedding_backward(G,idx,weight.shape[0],v['padding_idx'],v['scale_grad_by_freq']).data
        elif base=='gather':
            # Dense gather reverses to indexed addition, not indexed overwrite:
            # repeated token IDs must accumulate all their upstream credit.
            if v.get('sparse_grad',False):raise TapeError('SPARSE_GATHER_GRADIENT_NOT_ADMITTED')
            index=arr('index');rank=len(x.shape)
            if rank==0 or len(index.shape)!=rank:raise TapeError('GATHER_VJP_NONSCALAR_MATCHING_RANK_REQUIRED')
            dim=v['dim']%rank;result=[0.]*len(x.data)
            dtype=self.nodes[ids['self']].dtype
            for coordinate,upstream in zip(k.coords(outshape),g):
                target=list(coordinate);target[dim]=int(index.at(coordinate))
                flat=0
                for item,size in zip(target,x.shape):
                    if not 0<=item<size:raise TapeError('GATHER_VJP_INDEX_OUT_OF_BOUNDS')
                    flat=flat*size+item
                # Match sequential native CPU scatter-add rounding at each
                # accumulation, including cancellation after repeated indices.
                result[flat]=words.decode(words.encode_scalar(result[flat]+upstream,dtype),dtype)
            ans['self']=tuple(result)
        elif base=='convolution':
            if v['transposed'] or any(v['output_padding']):raise TapeError('TRANSPOSED_CONVOLUTION_VJP_NOT_ADMITTED')
            gx,gw,gb=C.convolution_backward(G,arr('input'),arr('weight'),v['stride'],v['padding'],v['dilation'],v['groups'],(True,True,True))
            ans['input']=gx.data;ans['weight']=gw.data
            if type(ids.get('bias')) is int:ans['bias']=gb.data
        elif base=='stack':
            dim=v['dim']%len(outshape)
            for j,ident in enumerate(ids['tensors']):
                a=self.array(ident)
                ans[('tensors',j)]=tuple(G.at(ix[:dim]+(j,)+ix[dim:]) for ix in k.coords(a.shape))
        elif base=='cat':
            dim=v['dim']%len(outshape);cursor=0
            for j,ident in enumerate(ids['tensors']):
                a=self.array(ident)
                ans[('tensors',j)]=tuple(G.at(ix[:dim]+(ix[dim]+cursor,)+ix[dim+1:]) for ix in k.coords(a.shape));cursor+=a.shape[dim]
        elif base=='cumsum':
            dim=v['dim']%len(x.shape)
            ans['self']=tuple(sum(G.at(ix[:dim]+(j,)+ix[dim+1:]) for j in range(ix[dim],x.shape[dim])) for ix in k.coords(x.shape))
        elif base=='constant_pad_nd':
            pads=v['pad'];before=[0]*len(x.shape)
            for j in range(len(pads)//2):before[-1-j]=pads[j*2]
            vals=[]
            for ix in k.coords(x.shape):
                out=tuple(i+b for i,b in zip(ix,before))
                vals.append(G.at(out) if all(0<=i<n for i,n in zip(out,outshape)) else 0.)
            ans['self']=tuple(vals)
        elif base in ('triu','tril'):
            diag=v.get('diagonal',0)
            ans['self']=tuple(G.at(ix) if (ix[-1]-ix[-2]>=diag if base=='triu' else ix[-1]-ix[-2]<=diag) else 0. for ix in k.coords(x.shape))
        elif base=='masked_fill':
            mask=arr('mask');put('self',tuple(0. if C.bat(mask,ix) else z for ix,z in zip(k.coords(outshape),g)))
        elif base in ('_softmax','_log_softmax'):
            dim=v['dim'];p=C.softmax(x,dim,log=False)
            summed=C.reduce(ew(lambda a,b:a*b,G,p) if base=='_softmax' else G,[dim],keepdim=True)
            ans['self']=(ew(lambda a,b,c:b*(a-c),G,p,summed) if base=='_softmax' else ew(lambda a,b,c:a-b*c,G,p,summed)).data
        else:raise TapeError('ACTIVE_VJP_NOT_IMPLEMENTED:'+name)
        return ans
    def backward(self,output,*,gradient=None):
        if not self.closed or self.failed or self.backward_done:raise TapeError('CLOSED_SUCCESSFUL_UNUSED_TAPE_REQUIRED')
        if threading.get_ident()!=self.owner_thread:raise TapeError('TAPE_THREAD_MISMATCH')
        self.verify_leaves()
        with torch._C._DisableTorchDispatch():output_node=self.read(output)
        out=self.nodes[output_node]
        if not out.active:raise TapeError('OUTPUT_HAS_NO_DIFFERENTIABLE_PATH')
        if gradient is None:
            if len(out.data)!=1:raise TapeError('NONSCALAR_REQUIRES_GRADIENT')
            seed=(1.0,)
        else:
            if not is_tensor(gradient) or tuple(gradient.shape)!=out.shape:raise TapeError('GRADIENT_SHAPE')
            seed=self.values(gradient)
        adjoints={output_node:seed};self.emit('PY_BACKWARD_BEGIN',output_node=output_node,nodes=len(self.nodes),native_autograd=False)
        try:
            for index in range(len(self.nodes)-1,-1,-1):
                n=self.nodes[index];g=adjoints.get(index)
                if g is None or not n.active:continue
                self.emit('PY_BACKWARD_NODE',node=index,operation=n.op,parents=list(n.parents),gradient_shape=list(n.shape),
                    **({'gradient':[self.encode_number(x) for x in g]} if self.record_values else {}))
                if n.backward is None:continue
                results=n.backward(g)
                if len(results)!=len(n.parents):raise TapeError('VJP_ARITY')
                for parent,values in zip(n.parents,results):
                    p=self.nodes[parent]
                    if not p.active:continue
                    if len(values)!=len(p.data):raise TapeError('VJP_SHAPE')
                    old=adjoints.get(parent)
                    summed=values if old is None else tuple(a+b for a,b in zip(old,values))
                    rounded=tuple(words.decode(words.encode_scalar(z,p.dtype),p.dtype) for z in summed)
                    if any(not math.isfinite(z) for z in rounded):
                        self.emit('PY_NONFINITE_GRADIENT',node=parent,values=[float(z).hex() for z in rounded]);raise TapeError('NONFINITE_GRADIENT_RECORDED_NOT_REPAIRED')
                    self.emit('PY_BACKWARD_ACCUMULATE',node=parent,from_node=index,
                        **({'incoming':[self.encode_number(x) for x in values],
                            'previous':None if old is None else [self.encode_number(x) for x in old],
                            'rounded':[self.encode_number(x) for x in rounded]} if self.record_values else {}))
                    adjoints[parent]=rounded
            for ident,(t,node,positions) in self.leaves.items():
                g=adjoints.get(node)
                if g is not None:self.gradients[ident]=tuple(g[p] for p in positions)
            self.verify_leaves()
            self.backward_done=True
            self.emit('PY_BACKWARD_END',leaf_gradients=len(self.gradients),native_autograd=False)
            return self
        except BaseException:
            self.failed=True;self.emit('PY_BACKWARD_FAILED');raise
    def verify_leaves(self):
        if self.frozen:
            try:self.frozen.verify_all()
            except BaseException:self.failed=True;raise
        for t,index,positions in self.leaves.values():
            self.reject_hooks(t)
            expected=self.nodes[index]
            if t.dtype is not expected.dtype or self.positions(t)!=positions:
                raise TapeError('LEAF_METADATA_CHANGED_AFTER_FORWARD')
            actual=self.values(t)
            if not all(exact_same(a,expected.data[p]) for a,p in zip(actual,positions)):
                raise TapeError('LEAF_VALUES_CHANGED_AFTER_FORWARD')
    def gradient(self,t):
        if not self.backward_done or self.failed:raise TapeError('NO_SUCCESSFUL_BACKWARD')
        data=self.gradients.get(id(t))
        if data is None:return None
        with torch._C._DisableTorchDispatch(),torch.no_grad():
            result=k.allocate(tuple(t.shape),dtype=t.dtype,memory_format=torch.contiguous_format)
            view=k.raw(result);width=result.element_size()
            for j,value in enumerate(data):view[j*width:(j+1)*width]=words.encode_scalar(value,t.dtype)
            return result
    def assign_gradients(self):
        if not self.backward_done or self.failed:raise TapeError('NO_SUCCESSFUL_BACKWARD')
        self.verify_leaves()
        prepared=[(t,self.gradient(t)) for t,_,_ in self.leaves.values()]
        self.verify_leaves()
        for t,g in prepared:t.grad=g
