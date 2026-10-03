"""Original text bodies -> Python tape backward -> readable save -> new process.

Synthetic four-layer data/config, not an actual True or 9B checkpoint. The
original code/data boundaries remain explicit. No native backward is called in
candidate steps; the native reference is separately labelled and measured.
"""
from __future__ import annotations
import argparse,contextlib,copy,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.text_model_entry import TextEntry,GenerationSession,model_digest,candidate
from tools.run_text_lifecycle import rng_state,set_rng,draw,data_hash
from tools.native_component_fixture import SOURCE
from xiaomai_storage.python_autodiff import PythonTape
from xiaomai_storage.python_optimizer import PythonSGD
import xiaomai_storage.python_autodiff as ad
from xiaomai_storage.allocation import AllocationScope
import xiaomai_storage.allocation as alloc
from xiaomai_storage import literal_state
from xiaomai_storage.kernels import logical_bytes
from xiaomai_storage.block_trace import BlockTrace,verify as verify_trace
from xiaomai_storage.line_trace import LineTrace

ATOL=5e-6;RTOL=3e-4

def sources():
    paths=[Path(__file__).resolve(),SOURCE,Path(candidate.__file__),Path(candidate.core.__file__),
           Path(candidate.training_ops.__file__),Path(candidate.streaming_mm.__file__)]
    paths+=list((ROOT/'xiaomai_storage').glob('*.py'))
    # Exclude observer implementation itself, not model/runtime data.
    return sorted({p.resolve() for p in paths if p.name not in ('line_trace.py','block_trace.py','compressed_trace.py','backend.py','code_identity.py')})

def source_pin():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}
def tsha(t):return hashlib.sha256(logical_bytes(t)).hexdigest()

def compare_values(a,b,atol,rtol):
    if a.shape!=b.shape or a.dtype is not b.dtype:raise RuntimeError('COMPARISON_SHAPE_OR_DTYPE')
    from xiaomai_storage.numeric_words import decode
    av,bv=logical_bytes(a),logical_bytes(b);w=a.element_size();largest=0.;reference=0.
    for j in range(0,len(av),w):
        x,y=decode(av[j:j+w],a.dtype),decode(bv[j:j+w],b.dtype)
        if not (__import__('math').isfinite(x) and __import__('math').isfinite(y)):
            raise RuntimeError('NONFINITE_COMPARISON_NOT_PASSED')
        difference=abs(x-y);largest=max(largest,difference);reference=max(reference,abs(y))
        if difference>atol+rtol*abs(y):raise RuntimeError('NUMERICAL_COMPARISON_FAILED_AT:'+str(j//w))
    return {'max_abs':largest,'oracle_max_abs':reference}

def progress(root,name,state):
    row={'phase':name,'state':state,'pid':os.getpid(),'time':time.time()}
    with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(name,state,flush=True)

@contextlib.contextmanager
def forbid_native_backward():
    old=[torch.autograd.grad,torch.autograd.backward,torch.Tensor.backward,torch.optim.SGD.step]
    def forbidden(*a,**k):raise RuntimeError('NATIVE_AUTOGRAD_CALLED_IN_CANDIDATE')
    torch.autograd.grad=forbidden;torch.autograd.backward=forbidden;torch.Tensor.backward=forbidden;torch.optim.SGD.step=forbidden
    try:yield
    finally:torch.autograd.grad,torch.autograd.backward,torch.Tensor.backward,torch.optim.SGD.step=old

def phase(root,name,fn,detail,*,native=False):
    start=time.time();progress(root,name,'BEGIN');path=root/name
    with BlockTrace(path,run_id='v087-training',phase=name) as log:
        tracer=contextlib.nullcontext() if detail=='calls' else LineTrace(sources(),log.emit,opcodes=detail=='instructions')
        with tracer:
            if native:result=fn(log.emit)
            else:
                with forbid_native_backward(),AllocationScope(strategy='python-buffer',sink=log.emit) as allocation,candidate.PythonReferenceBackend() as backend:
                    result=fn(log.emit)
                    log.emit('BACKEND_COUNTS',counts={label:sum(r['implementation']==label for r in backend.calls) for label in {r['implementation'] for r in backend.calls}},
                        native_forward_boundaries=True,allocation_calls=allocation.count,allocated_bytes=allocation.total)
    checked=verify_trace(path,expected_seal_sha256=log.receipt['seal_sha256'])
    if detail!='calls' and checked['kinds'].get('PY_EXECUTION',0)==0:raise RuntimeError('MISSING_REQUESTED_LINES')
    r={'status':'PHASE_PASS','trace_path':str(path),'detail':detail,'seal_sha256':log.receipt['seal_sha256'],
       'trace':checked,'elapsed_seconds':time.time()-start,'native_reference':native}
    progress(root,name,'PASS');return result,r

def training_step(model,opt,ids,sink):
    model.zero_grad(set_to_none=True)
    with PythonTape(sink=sink,record_values=True) as tape:
        output=model(ids);loss=output.square().mean()
    if loss.requires_grad or loss.grad_fn is not None:raise RuntimeError('NATIVE_GRAPH_PRESENT')
    tape.backward(loss);tape.assign_gradients()
    catalog=[{'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'leaf_node':tape.leaves[id(p)][1],
              'gradient_sha256':tsha(p.grad)} for name,p in model.named_parameters() if id(p) in tape.leaves]
    sink('PY_PARAMETER_CATALOG',parameters=catalog)
    before=model_digest(model);previous_sink=opt.sink;opt.sink=sink
    try:opt.step()
    finally:opt.sink=previous_sink
    if model_digest(model)==before:raise RuntimeError('NO_UPDATE')
    return {'loss':float(loss).hex(),'output':tsha(output),'nodes':len(tape.nodes),'saved_elements':tape.saved,
            'gradients':catalog,'native_autograd':False,'parameter_count':sum(p.numel() for p in model.parameters())}

def child(args):
    phases={}
    package,phases['restore']=phase(args.out,'restore',lambda sink:literal_state.load(args.child,expected_manifest_sha256=args.manifest_sha,sink=sink),args.detail)
    if package['source_pins']!=source_pin():raise RuntimeError('SOURCE_PIN_CHANGED')
    model=TextEntry();opt=PythonSGD(model.parameters(),lr=.001,momentum=.9)
    def bind(sink):
        model.load_state_dict(package['model']);opt.load_state_dict(package['optimizer']);set_rng(package['rng'])
        return GenerationSession.from_data(model,package['session'])
    session,phases['bind']=phase(args.out,'bind',bind,args.detail)
    probe=draw()
    generated,phases['generate']=phase(args.out,'generate',lambda sink:session.feed(package['next_ids'],sink=sink),args.detail)
    step,phases['train']=phase(args.out,'train',lambda sink:training_step(model,opt,package['next_ids'],sink),args.detail)
    result={'model':model_digest(model),'optimizer':data_hash(opt.state_dict()),'rng_probe':probe,
            'generated':{'logits':tsha(generated[0]),'token':generated[1]},'next_step':step,
            'phases':phases,'pid':os.getpid(),'native_autograd_in_candidate':False}
    (args.out/'CHILD_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def parent(args):
    out=args.out;start=time.time();phases={}
    random.seed(872);np.random.seed(872);torch.manual_seed(872)
    model=TextEntry();opt=PythonSGD(model.parameters(),lr=.001,momentum=.9)
    ids=torch.tensor([[1,2]]);next_ids=torch.tensor([[3,4]])
    initial={n:t.detach().clone() for n,t in model.state_dict().items()};initial_rng=rng_state();initial_opt=copy.deepcopy(opt.state_dict())
    identity={n:id(p) for n,p in model.named_parameters()};initial_digest=model_digest(model)
    def native_reference(sink):
        output=model(ids);output.square().mean().backward()
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
        sink('NATIVE_ORACLE_ONLY',autograd_used=True,not_candidate=True)
        return output.detach(),gradients
    (native,grads),phases['native_oracle']=phase(out,'native_oracle',native_reference,args.detail,native=True)
    model.zero_grad(set_to_none=True)
    def compare(sink):
        with PythonTape(sink=sink,record_values=True) as t:
            output=model(ids);loss=output.square().mean()
        if loss.grad_fn is not None:raise RuntimeError('NATIVE_GRAPH_PRESENT')
        t.backward(loss);t.assign_gradients();rows=[]
        compare_values(output,native,atol=2e-5,rtol=2e-5)
        for name,p in model.named_parameters():
            rows.append({'name':name,**compare_values(p.grad,grads[name],atol=ATOL,rtol=RTOL)})
        return output.detach(),rows
    (baseline,comparisons),phases['python_backward_compare']=phase(out,'python_backward_compare',compare,args.detail)
    # Uninstrumented candidate outputs and gradients are a separate control.
    with candidate.PythonReferenceBackend(),PythonTape() as control:
        c=model(ids);closs=c.square().mean()
    control.backward(closs)
    if not torch.equal(c,baseline):raise RuntimeError('OBSERVATION_OUTPUT_DIFFERENCE')
    for n,p in model.named_parameters():
        if not torch.equal(control.gradient(p),p.grad):raise RuntimeError('OBSERVATION_GRADIENT_DIFFERENCE')
    chosen=model.language_model.layers[0].input_layernorm.weight;saved=chosen.detach().clone()
    def intervene(sink):
        with torch.no_grad():chosen.add_(.125)
        return model(ids)
    changed,phases['intervene']=phase(out,'intervene',intervene,args.detail)
    if torch.equal(changed,baseline):raise RuntimeError('NO_INTERVENTION_EFFECT')
    def revert(sink):
        with torch.no_grad():chosen.copy_(saved)
        return model(ids)
    restored,phases['revert']=phase(out,'revert',revert,args.detail)
    if not torch.equal(restored,baseline):raise RuntimeError('REVERT_MISMATCH')
    old_session=GenerationSession(model)
    _,phases['old_cache']=phase(out,'old_cache',lambda sink:old_session.feed(ids,sink=sink),args.detail)
    update,phases['update']=phase(out,'update',lambda sink:training_step(model,opt,ids,sink),args.detail)
    def stale(sink):
        try:old_session.feed(next_ids)
        except ValueError as exc:
            if 'STALE_CACHE' not in str(exc):raise
            sink('EXPECTED_CACHE_REJECTION',reason=str(exc));return True
        raise RuntimeError('STALE_CACHE_ACCEPTED')
    _,phases['stale_cache']=phase(out,'stale_cache',stale,args.detail)
    session=GenerationSession(model)
    _,phases['new_cache']=phase(out,'new_cache',lambda sink:session.feed(ids,sink=sink),args.detail)
    cp=out/'READABLE_CHECKPOINT'
    checkpoint,phases['save']=phase(out,'save',lambda sink:literal_state.save({'source_pins':source_pin(),'model':model.state_dict(),
        'optimizer':opt.state_dict(),'session':session.data(),'rng':rng_state(),'next_ids':next_ids},cp,sink=sink),args.detail)
    probe=draw()
    generated,phases['next_generation']=phase(out,'next_generation',lambda sink:session.feed(next_ids,sink=sink),args.detail)
    step,phases['next_update']=phase(out,'next_update',lambda sink:training_step(model,opt,next_ids,sink),args.detail)
    expected={'model':model_digest(model),'optimizer':data_hash(opt.state_dict()),'rng_probe':probe,
              'generated':{'logits':tsha(generated[0]),'token':generated[1]},'next_step':step}
    childout=out/'child';childout.mkdir()
    command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',str(cp),'--manifest-sha',checkpoint['manifest_sha256'],
             '--out',str(childout),'--detail',args.detail]
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    with (out/'child.stdout').open('wb') as so,(out/'child.stderr').open('wb') as se:
        child_process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=so,stderr=se);exitcode=child_process.wait()
    if exitcode:raise RuntimeError('CHILD_FAILED:'+str(exitcode))
    actual=json.loads((childout/'CHILD_RESULT.json').read_text())
    if any(actual[k]!=v for k,v in expected.items()):raise RuntimeError('CHILD_NUMERICAL_STATE_MISMATCH')
    literal_state.verify(cp,expected_manifest_sha256=checkpoint['manifest_sha256'])
    def rollback(sink):
        model.load_state_dict(initial);opt.load_state_dict(initial_opt);model.zero_grad(set_to_none=True);set_rng(initial_rng)
        return model(ids)
    restored,phases['rollback']=phase(out,'rollback',rollback,args.detail)
    if not torch.equal(restored,baseline) or model_digest(model)!=initial_digest:raise RuntimeError('ROLLBACK_MISMATCH')
    if identity!={n:id(p) for n,p in model.named_parameters()}:raise RuntimeError('OBJECT_IDENTITY_CHANGED')
    if data_hash(opt.state_dict())!=data_hash(initial_opt) or data_hash(rng_state())!=data_hash(initial_rng):raise RuntimeError('ROLLBACK_STATE_MISMATCH')
    result={'status':'PYTHON_BACKWARD_TEXT_LITERAL_LIFECYCLE_PASS','parameter_count':update['parameter_count'],
       'source_receipt':model.receipt,'source_pins':source_pin(),'native_vs_candidate_gradients':comparisons,
       'gradient_tolerance':{'atol':ATOL,'rtol':RTOL},'native_python_output_max_abs':float((native-baseline).abs().max()),
       'candidate_has_native_autograd_graph':False,'candidate_native_backward_calls_allowed':False,
       'observer_on_off_output_and_gradients_bitwise_equal':True,'python_reverse_scheduling':True,
       'python_buffer_allocations':True,'native_tensor_binding':True,'native_optimizer_used_in_candidate':False,'python_optimizer':'PythonSGD_v1',
       'intervention_reverted':True,'stale_cache_rejected':True,'new_process_next_generation_update_equal':True,
       'rollback_bitwise_equal':True,'parameter_object_identity_preserved':True,'optimizer_rng_restored':True,
       'checkpoint':checkpoint,'detail':args.detail,'phases':phases,'child_phases':actual['phases'],
       'child_pid':child_process.pid,'child_command':command,'child_exit_code':exitcode,'elapsed_seconds':time.time()-start,
       'real_weights':False,'current_true_restored':False,'GPU_tested':False,'entire_xiaomai_python_conversion_complete':False}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','parameter_count','elapsed_seconds','detail')}))

def main():
    if sys.flags.optimize:raise RuntimeError('VERIFIER_REQUIRES_ASSERTIONS_ENABLED')
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--detail',choices=['calls','lines','instructions'],default='lines')
    p.add_argument('--child',type=Path);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.child:
        if not a.out.is_dir() or not a.manifest_sha:raise RuntimeError('CHILD_CUSTODY_REQUIRED')
        child(a)
    else:a.out.mkdir(parents=True,exist_ok=False);parent(a)
if __name__=='__main__':main()
