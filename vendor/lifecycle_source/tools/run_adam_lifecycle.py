"""PythonTape + grouped AdamW + clipping + literal checkpoint + real child process.

Uses the pinned original text-model bodies with synthetic reduced configuration.
All candidate backward/optimizer/clip entrypoints forbid their native counterpart.
Detailed traces are optional; the selected scope is always reported explicitly.
"""
from __future__ import annotations
import argparse,contextlib,copy,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from tools.text_model_entry import TextEntry,GenerationSession,model_digest,candidate
from tools.run_text_lifecycle import rng_state,set_rng,draw,data_hash
from tools.run_python_training_lifecycle import compare_values
from xiaomai_storage.python_autodiff import PythonTape
from xiaomai_storage.adam_optimizer import PythonAdamW,sha
from xiaomai_storage.gradient_clip import clip_grad_norm_python
from xiaomai_storage.literal_state import save,load as load_state,verify as verify_state
from xiaomai_storage.block_trace import BlockTrace,verify as verify_trace
from xiaomai_storage.line_trace import LineTrace


def source_pins():
    roots=('xiaomai_storage','integration','tools','upstream','reference_support')
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for group in roots for p in (ROOT/group).rglob('*.py') if '__pycache__' not in p.parts}


def optimizer(model):
    names=list(model.named_parameters());norms=[x for x in names if 'norm' in x[0]];others=[x for x in names if 'norm' not in x[0]]
    return PythonAdamW([{'params':norms,'group_name':'norm','lr':.001,'weight_decay':0.,'betas':(.8,.95)},
                       {'params':others,'group_name':'projection','lr':.0005,'weight_decay':.01,'amsgrad':True}])


@contextlib.contextmanager
def forbid_native_learning():
    targets=[(torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward'),
             (torch.optim.Adam,'step'),(torch.optim.AdamW,'step'),(torch.optim.SGD,'step'),
             (torch.nn.utils,'clip_grad_norm_')]
    old=[(obj,name,getattr(obj,name)) for obj,name in targets]
    def forbidden(*a,**k):raise RuntimeError('NATIVE_LEARNING_CALLED_IN_CANDIDATE')
    for obj,name,_ in old:setattr(obj,name,forbidden)
    try:yield
    finally:
        for obj,name,value in old:setattr(obj,name,value)


def phase(root,name,fn,detail='calls',*,native=False,backend=True):
    started=time.time()
    with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'phase':name,'state':'BEGIN','pid':os.getpid(),'time':started})+'\n');f.flush();os.fsync(f.fileno())
    with BlockTrace(root/name,run_id='v088-native-learner-integration',phase=name) as log:
        paths=[ROOT/'xiaomai_storage'/n for n in ('adam_optimizer.py','gradient_clip.py','evidence_update.py','data_snapshot.py','literal_state.py')]
        paths+=[Path(__file__).resolve(),ROOT/'tools/run_evidence_learner_lifecycle.py',ROOT/'upstream/evidence_learning.py',ROOT/'upstream/intrinsic_core.py',ROOT/'upstream/intrinsic_neural.py']
        tracer=LineTrace(paths,log.emit,opcodes=detail=='instructions') if detail!='calls' else contextlib.nullcontext()
        with tracer:
            if native:result=fn(log.emit)
            else:
                with forbid_native_learning():
                    ctx=candidate.PythonReferenceBackend() if backend else contextlib.nullcontext()
                    with ctx:result=fn(log.emit)
    checked=verify_trace(root/name,expected_seal_sha256=log.receipt['seal_sha256'])
    if detail!='calls' and checked['kinds'].get('PY_EXECUTION',0)==0:raise RuntimeError('REQUESTED_TRACE_MISSING')
    receipt={'phase':name,'status':'PASS','path':str(root/name),'detail':detail,'seal_sha256':log.receipt['seal_sha256'],
             'trace':checked,'elapsed_seconds':time.time()-started,'native_reference':native,
             'line_scope':'optimizer, clipping, snapshot helpers, original evidence/graft/control, adapters and runners; not all model/library lines'}
    with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'phase':name,'state':'PASS','pid':os.getpid(),'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
    return result,receipt


def loss_and_gradients(model,ids,target,sink):
    model.zero_grad(set_to_none=True)
    with PythonTape(sink=sink,record_values=True) as tape:
        out=model(ids)
        loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),target.reshape(-1))
    if loss.grad_fn is not None:raise RuntimeError('NATIVE_GRAPH_PRESENT')
    tape.backward(loss);tape.assign_gradients()
    return out,loss,tape


def train(model,opt,ids,target,sink):
    out,loss,tape=loss_and_gradients(model,ids,target,sink)
    norm=clip_grad_norm_python(list(model.named_parameters()),1.,sink=sink)
    old=opt.sink;opt.sink=sink
    try:opt.step()
    finally:opt.sink=old
    return {'logits':sha(out),'loss':float(loss).hex(),'norm':norm.hex(),
            'node_count':len(tape.nodes),'optimizer':data_hash(opt.state_dict()),
            'model':model_digest(model)}


def child(a):
    phases={}
    payload,phases['restore']=phase(a.out,'restore',lambda sink:load_state(a.child,expected_manifest_sha256=a.manifest_sha,sink=sink),a.detail)
    if payload['sources']!=source_pins():raise RuntimeError('SOURCE_PIN_CHANGED')
    model=TextEntry();opt=optimizer(model)
    def bind(sink):
        model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);set_rng(payload['rng'])
        return GenerationSession.from_data(model,payload['session'])
    session,phases['bind']=phase(a.out,'bind',bind,a.detail)
    probe=draw()
    (logits,token),phases['next_generation']=phase(a.out,'next_generation',lambda sink:session.feed(payload['next_ids'],sink=sink),a.detail)
    updated,phases['next_update']=phase(a.out,'next_update',lambda sink:train(model,opt,payload['next_ids'],payload['next_targets'],sink),a.detail)
    result={'generation':{'logits':sha(logits),'token':token},'update':updated,'rng':probe,'phases':phases,'pid':os.getpid()}
    (a.out/'CHILD_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def parent(a):
    start=time.time();root=a.out;phases={}
    torch.manual_seed(888);random.seed(888);np.random.seed(888)
    model=TextEntry();opt=optimizer(model)
    ids=torch.tensor([[1,2]]);targets=torch.tensor([[2,3]])
    next_ids=torch.tensor([[3,4]]);next_targets=torch.tensor([[4,5]])
    original=copy.deepcopy(model.state_dict());before_opt=opt.state_dict();before_rng=rng_state()
    original_ids={n:id(p) for n,p in model.named_parameters()};original_digest=model_digest(model)
    def reference(sink):
        model.zero_grad(set_to_none=True);out=model(ids);loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),targets.reshape(-1));loss.backward()
        return out.detach(),{n:p.grad.detach().clone() for n,p in model.named_parameters()}
    (native,gradients),phases['native_reference']=phase(root,'native_reference',reference,native=True)
    def compare(sink):
        out,loss,tape=loss_and_gradients(model,ids,targets,sink)
        rows={n:compare_values(p.grad,gradients[n],atol=5e-6,rtol=3e-4) for n,p in model.named_parameters()}
        return out.detach(),rows
    (baseline,rows),phases['python_ce_backward']=phase(root,'python_ce_backward',compare,a.detail)
    comparison=compare_values(baseline,native,atol=2e-5,rtol=2e-5)
    observed_gradients={n:sha(p.grad) for n,p in model.named_parameters()}
    # Same candidate without a sink is a separate observation control.
    with candidate.PythonReferenceBackend(),forbid_native_learning():
        check,_,control=loss_and_gradients(model,ids,targets,None)
    if sha(check)!=sha(baseline):raise RuntimeError('OBSERVATION_OUTPUT_CHANGED')
    if observed_gradients!={n:sha(p.grad) for n,p in model.named_parameters()}:raise RuntimeError('OBSERVATION_GRADIENT_CHANGED')
    for n,p in model.named_parameters():
        compare_values(p.grad,gradients[n],atol=5e-6,rtol=3e-4)
    chosen=model.language_model.layers[0].input_layernorm.weight;saved=chosen.detach().clone()
    def intervention(sink):
        with torch.no_grad():chosen.add_(.125)
        return model(ids)
    modified,phases['intervene']=phase(root,'intervene',intervention,a.detail)
    if sha(modified)==sha(baseline):raise RuntimeError('NO_INTERVENTION_EFFECT')
    def revert(sink):
        with torch.no_grad():chosen.copy_(saved)
        return model(ids)
    reverted,phases['revert']=phase(root,'revert',revert,a.detail)
    if sha(reverted)!=sha(baseline):raise RuntimeError('INTERVENTION_NOT_REVERTED')
    old=GenerationSession(model)
    _,phases['build_old_cache']=phase(root,'build_old_cache',lambda sink:old.feed(ids,sink=sink),a.detail)
    first_update,phases['adam_update']=phase(root,'adam_update',lambda sink:train(model,opt,ids,targets,sink),a.detail)
    if model_digest(model)==original_digest:raise RuntimeError('NO_UPDATE')
    try:old.feed(next_ids)
    except ValueError as e:
        if 'STALE_CACHE' not in str(e):raise
    else:raise RuntimeError('STALE_CACHE_NOT_REJECTED')
    session=GenerationSession(model)
    _,phases['rebuild_cache']=phase(root,'rebuild_cache',lambda sink:session.feed(ids,sink=sink),a.detail)
    cp=root/'READABLE_CHECKPOINT'
    saved_cp,phases['save']=phase(root,'save',lambda sink:save({'sources':source_pins(),'model':model.state_dict(),
        'optimizer':opt.state_dict(),'rng':rng_state(),'session':session.data(),'next_ids':next_ids,'next_targets':next_targets},cp,sink=sink),a.detail)
    probe=draw()
    (logits,token),phases['parent_next_generation']=phase(root,'parent_next_generation',lambda sink:session.feed(next_ids,sink=sink),a.detail)
    next_update,phases['parent_next_update']=phase(root,'parent_next_update',lambda sink:train(model,opt,next_ids,next_targets,sink),a.detail)
    expected={'generation':{'logits':sha(logits),'token':token},'update':next_update,'rng':probe}
    childout=root/'child';childout.mkdir()
    cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',str(cp),'--manifest-sha',saved_cp['manifest_sha256'],
         '--out',str(childout),'--detail',a.detail]
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    with (root/'child.stdout').open('wb') as so,(root/'child.stderr').open('wb') as se:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=so,stderr=se);code=process.wait()
    if code:raise RuntimeError('CHILD_FAILED')
    got=json.loads((childout/'CHILD_RESULT.json').read_text())
    if any(got[n]!=v for n,v in expected.items()):raise RuntimeError('CHILD_STATE_DIFFERENCE')
    verify_state(cp,expected_manifest_sha256=saved_cp['manifest_sha256'])
    def rollback(sink):
        model.load_state_dict(original);opt.load_state_dict(before_opt);set_rng(before_rng);model.zero_grad(set_to_none=True)
        return model(ids)
    restored,phases['rollback']=phase(root,'rollback',rollback,a.detail)
    if sha(restored)!=sha(baseline) or model_digest(model)!=original_digest:raise RuntimeError('ROLLBACK_VALUES')
    if {n:id(p) for n,p in model.named_parameters()}!=original_ids:raise RuntimeError('ROLLBACK_IDENTITY')
    if data_hash(opt.state_dict())!=data_hash(before_opt) or data_hash(rng_state())!=data_hash(before_rng):raise RuntimeError('ROLLBACK_OPTIMIZER_RNG')
    result={'status':'PYTHON_ADAMW_MULTI_GROUP_LITERAL_LIFECYCLE_PASS','parameter_count':sum(p.numel() for p in model.parameters()),
      'groups':[(g['group_name'],len(g['params'])) for g in opt.param_groups],
      'native_python_forward':comparison,'gradient_comparisons':rows,'gradient_tolerance':{'atol':5e-6,'rtol':3e-4},
      'native_learning_calls_forbidden':True,'trace_on_off_output_and_gradients_bitwise_equal':True,'stale_cache_rejected':True,'new_process_next_generation_update_identical':True,
      'rollback_values_optimizer_rng':True,'parameter_objects_preserved':True,'phases':phases,'child_phases':got['phases'],
      'checkpoint':saved_cp,'child_pid':process.pid,'child_exit_code':code,'child_command':cmd,
      'elapsed_seconds':time.time()-start,'detail':a.detail,'real_weights':False,'current_true':False,'GPU':False,
      'complete_python_conversion':False,'references':model.receipt}
    (root/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','parameter_count','groups','elapsed_seconds','detail')}))


def main():
    if sys.flags.optimize:raise RuntimeError('VERIFICATION_REQUIRES_ASSERTIONS')
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--detail',choices=['calls','lines','instructions'],default='calls')
    p.add_argument('--child',type=Path);p.add_argument('--manifest-sha');a=p.parse_args()
    if a.child:
        if not a.manifest_sha or not a.out.is_dir():raise RuntimeError('CHILD_CUSTODY_REQUIRED')
        child(a)
    else:a.out.mkdir(parents=True,exist_ok=False);parent(a)
if __name__=='__main__':main()
