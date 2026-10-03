"""Head/value-column independent chunk VJP with checkpoint recomputation.

Always retains C-sized triangular solves. Splitting value columns is legal
because no value column affects another; q/k/g/beta adjoints are summed over
columns. Segment final-state cotangents propagate backwards across chunks.
Never substitutes a per-token recurrent kernel. Native boundaries are stated.
"""
import math
import time
import numpy as np
from .chunk_vjp import ChunkDomainError, ChunkResult, NAMES, _sha, _validate, chunk_vjp


def graph_capacity(T, K, V, C=64, normalise=False, compact=True):
    """Exact operation counts for B=H=1; constants assume K>1 (K=1 adjusted)."""
    if min(T,K,V,C)<=0 or C>128: raise ChunkDomainError('CAPACITY_SHAPE')
    chunks=math.ceil(T/C)
    leaf=T*(2*K+V+2)+K*V
    norm_nodes=T*2*(3*K+3) if normalise else 0
    norm_edges=T*2*(6*K+5) if normalise else 0
    strict=C*(C-1)//2; lower=C*(C+1)//2
    tri_nodes=sum(i*(2*i+2) for i in range(1,C))
    tri_edges=sum(i*(4*i+3) for i in range(1,C))
    mm_nodes=2*C*C+4*C*V+C*K+K*V
    mm_edges=2*(3*C*C*K+2*C*C*V+3*C*K*V)
    element_nodes=(2*C+C*(2*K+V)+2*lower+2*strict+tri_nodes+C+
        C*(1+2*K)+C*V+C*C+C*(2+K)+1+2*K*V)
    element_edges=(3*C+2*C*(2*K+V)+3*lower+3*strict+tri_edges+2*C+
        C*(1+4*K)+2*C*V+2*C*C+C*(3+2*K)+1+4*K*V)
    nodes=2+(K!=1)+int(normalise)+leaf+norm_nodes+chunks*(element_nodes+mm_nodes)+T*V
    edges=norm_edges+chunks*(element_edges+mm_edges)+2*T*V
    if not compact:
        nodes += chunks*mm_edges
        edges += chunks*(mm_edges+mm_nodes)
    return {'nodes':int(nodes),'reverse_edges':int(edges),'tensor_input_elements':leaf,
        'contracted_dots':compact,'chunk_size':C,'chunks':chunks,
        'python_graph_estimated_bytes':int(nodes*192+edges*176),
        'memory_estimate_note':'conservative planning coefficients; not exact Python allocator/RSS measurement'}


def bounded_chunk_vjp(q,k,v,g,beta,*,initial=None,grad_output=None,grad_state=None,
                      normalise=False,chunk_size=64,value_tile=16,max_nodes=1_000_000,
                      max_edges=6_000_000,max_bytes=2*1024**3,max_seconds=110,
                      max_jobs=64,observer=None):
    started=time.perf_counter()
    initial,dy,ds=_validate(q,k,v,g,beta,initial,grad_output,grad_state,chunk_size,max_nodes)
    if type(normalise) is not bool or type(value_tile) is not int or value_tile<1:
        raise ChunkDomainError('BOUNDED_POLICY')
    if (type(max_seconds) not in (int,float) or not 0<max_seconds<=120 or
            type(max_bytes) is not int or not 1<=max_bytes<=2*1024**3 or
            type(max_jobs) is not int or not 1<=max_jobs<=4096):
        raise ChunkDomainError('BOUNDED_RESOURCE_POLICY')
    B,T,H,K=q.shape;V=v.shape[-1];C=chunk_size
    jobs=B*H*math.ceil(T/C)*math.ceil(V/value_tile)
    plan=graph_capacity(min(T,C),K,min(V,value_tile),C,normalise,True)
    # Includes a conservative runtime/oracle reserve, arrays, output/gradients,
    # checkpoints. This planning threshold is NOT a measured hard RSS guarantee.
    retained=4*(B*T*H*(2*K+2*V+2)+3*B*H*K*V+math.ceil(T/C)*B*H*K*V)
    estimated_peak=plan['python_graph_estimated_bytes']+retained+512*1024**2
    if plan['nodes']>max_nodes or plan['reverse_edges']>max_edges or estimated_peak>max_bytes:
        raise ChunkDomainError('BOUNDED_CAPACITY_REJECT:'+str({'plan':plan,'estimated_peak_bytes':estimated_peak}))
    if jobs>max_jobs: raise ChunkDomainError('BOUNDED_JOB_BUDGET_REJECT:'+str(jobs))
    originals=(q,k,v,g,beta,initial);before=tuple(_sha(a) for a in originals)
    output=np.empty_like(v);state=np.empty_like(initial)
    gradients={name:np.zeros(a.shape,np.float64) for name,a in zip(NAMES,originals)}
    records=[]
    def remaining():
        value=max_seconds-(time.perf_counter()-started)
        if value<=0: raise ChunkDomainError('BOUNDED_TOTAL_WALL_TIME_EXCEEDED')
        return value
    def emit(event):
        if observer is not None: observer(event)
    # Only imported for actual multi-chunk forward checkpoint creation.
    from xiaomai_step_kernels.chunk import chunk
    for b in range(B):
        for h in range(H):
            for col in range(0,V,value_tile):
                end=min(col+value_tile,V)
                starts=list(range(0,T,C))
                checkpoints=[initial[b:b+1,h:h+1,:,col:end].copy()]
                # A single chunk needs no extra forward replay. Multi-chunk
                # checkpoints use the pinned Python triangular chunk forward.
                for start in starts[:-1]:
                    remaining()
                    stop=min(start+C,T)
                    _,next_state,_=chunk(q[b:b+1,start:stop,h:h+1],k[b:b+1,start:stop,h:h+1],
                        v[b:b+1,start:stop,h:h+1,col:end],g[b:b+1,start:stop,h:h+1],
                        beta[b:b+1,start:stop,h:h+1],initial=checkpoints[-1],normalise=normalise,
                        chunk_size=C,mode='interpreted')
                    checkpoints.append(next_state)
                carry=ds[b:b+1,h:h+1,:,col:end].copy()
                for index in range(len(starts)-1,-1,-1):
                    start=starts[index];stop=min(start+C,T)
                    tag={'batch':b,'head':h,'start':start,'stop':stop,'value_columns':[col,end]}
                    result=chunk_vjp(q[b:b+1,start:stop,h:h+1],k[b:b+1,start:stop,h:h+1],
                        v[b:b+1,start:stop,h:h+1,col:end],g[b:b+1,start:stop,h:h+1],
                        beta[b:b+1,start:stop,h:h+1],initial=checkpoints[index],
                        grad_output=dy[b:b+1,start:stop,h:h+1,col:end],grad_state=carry,
                        normalise=normalise,chunk_size=C,max_nodes=max_nodes,
                        compact_dots=True,max_edges=max_edges,max_seconds=remaining(),
                        observer=None if observer is None else lambda e:emit({'job':tag,**e}))
                    output[b:b+1,start:stop,h:h+1,col:end]=result.output
                    if index==len(starts)-1: state[b:b+1,h:h+1,:,col:end]=result.state
                    for name in ('q','k','g','beta'):
                        target=gradients[name][b:b+1,start:stop,h:h+1]
                        incoming=result.gradients[name]
                        for position in np.ndindex(target.shape):
                            target[position]=float(target[position])+float(incoming[position])
                    gradients['v'][b:b+1,start:stop,h:h+1,col:end]=result.gradients['v']
                    carry=result.gradients['initial']
                    records.append({**tag,'nodes':result.receipt['nodes'],'reverse_edges':result.receipt['reverse_edges']})
                    del result
                gradients['initial'][b:b+1,h:h+1,:,col:end]=carry
                emit({'phase':'head_tile_return','head':h,'batch':b,'value_columns':[col,end]})
    gradients={name:a.astype(np.float32) for name,a in gradients.items()}
    if before!=tuple(_sha(a) for a in originals): raise ChunkDomainError('BOUNDED_INPUT_DRIFT')
    if not all(np.isfinite(a).all() for a in (output,state,*gradients.values())):
        raise ChunkDomainError('BOUNDED_NONFINITE_RETURN')
    return ChunkResult(output,state,gradients,{'algorithm':'CHUNK_TRIANGULAR_HEAD_VALUE_TILE_CHECKPOINT_VJP',
        'chunk_size':C,'normalise':normalise,'value_tile':value_tile,'jobs':jobs,'job_records':records,
        'single_job_capacity':plan,'estimated_peak_bytes':estimated_peak,'seconds':time.perf_counter()-started,
        'output_and_final_state_joint_cotangents':True,'native_backward_used':False,'native_blas_used':False,
        'recurrent_substitution':False,'actual_gpu':False,'real_weights_used':False,'full_model_integrated':False,
        'dtype':'CPU_NUMPY_FP32','precision':'FP32 forward; binary64 dot and reverse; FP32 per-job adjoints; binary64 cross-tile sum then FP32',
        'native_boundaries':['NumPy allocation/index/copy/cast and finite checks','libm exp/sqrt','pinned interpreted chunk checkpoint forward'],
        'full_dot_internal_scalar_trace':False,'memory_limit_is_preflight_estimate_not_hard_os_cap':True,
        'input_sha256':dict(zip(NAMES,before))})


def iter_head_vjps(q,k,v,g,beta,*,head_indices=(0,),initial=None,
                   grad_output=None,grad_state=None,max_seconds=110,**policy):
    """Yield real numeric outputs/gradients one selected head at a time.

    Default selects head zero explicitly, NOT the complete layer. Caller must
    route/accumulate gradients through upstream repeated q/k and its Tape.
    Total iterator lifetime shares the wall budget; previously yielded heads
    are partial work, never an automatic complete-model acceptance receipt.
    """
    if type(max_seconds) not in (int,float) or not 0<max_seconds<=120:
        raise ChunkDomainError('ITERATOR_WALL_BUDGET')
    initial,dy,ds=_validate(q,k,v,g,beta,initial,grad_output,grad_state,
        policy.get('chunk_size',64),policy.get('max_nodes',1_000_000))
    if (type(head_indices) not in (tuple,list) or not head_indices or
        any(type(h) is not int or not 0<=h<q.shape[2] for h in head_indices) or
        len(set(head_indices))!=len(head_indices)):
        raise ChunkDomainError('ITERATOR_HEAD_SELECTION')
    deadline=time.perf_counter()+max_seconds
    for h in head_indices:
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise ChunkDomainError('ITERATOR_TOTAL_WALL_TIME_EXCEEDED')
        result=bounded_chunk_vjp(q[:,:,h:h+1],k[:,:,h:h+1],v[:,:,h:h+1],g[:,:,h:h+1],beta[:,:,h:h+1],
            initial=initial[:,h:h+1],grad_output=dy[:,:,h:h+1],grad_state=ds[:,h:h+1],
            max_seconds=remaining,**policy)
        result.receipt['selected_head']=h
        result.receipt['requested_head_indices']=list(head_indices)
        result.receipt['upstream_qk_repeat_derivative_integrated']=False
        yield h,result
