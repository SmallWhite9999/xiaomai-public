"""Pinned research neural runtime -> Python training -> literal new-process resume.

This is a CPU integration witness with small synthetic weights and explicit
engineering losses. It is not autonomous research, the production credit/loss
policy, the current True checkpoint, a 9B run or a GPU acceptance.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from tools.integration_backend import load
from tools.research_fixture import build
from tools.run_adam_lifecycle import forbid_native_learning
from tools.run_python_training_lifecycle import compare_values
from tools.run_text_lifecycle import draw,data_hash
from xiaomai_storage.python_autodiff import PythonTape
from xiaomai_storage.research_adapter import PythonResearchAdapter
from xiaomai_storage import research_checkpoint as checkpoint
from xiaomai_storage.adam_optimizer import PythonAdamW
from xiaomai_storage.gradient_clip import clip_grad_norm_python
from xiaomai_storage.literal_state import save,load as load_state,verify as verify_state
from xiaomai_storage.block_trace import BlockTrace,verify as verify_trace
from xiaomai_storage.line_trace import LineTrace
from xiaomai_storage.kernels import logical_bytes

candidate=load()
BUILD=dict(seed=9028,width=4,slots=2,steps=1,vocab_size=17,nonzero_bridge=True)
FIRST_ROWS=[[2,3,4],[5,6]]
FIRST_EVENTS=['evidence:first','evidence:second']
NEXT_ROWS=[[2,3,4],[5,6],[8,9,10]]
NEXT_EVENTS=FIRST_EVENTS+['evidence:next']
FIRST_IDS=[[1,2,3]];NEXT_IDS=[[3,7,8]]
CANDIDATES=[[4,6],[8,10]]
SCOPE=['tools/run_research_lifecycle.py','xiaomai_storage/research_adapter.py','xiaomai_storage/research_checkpoint.py',
       'upstream/research_native/experiments/ornith_problem_formation_v1/research_neural.py',
       'upstream/research_native/experiments/ornith_research_credit_v1/baseline/research_primitives.py']

def tensor_sha(t):return hashlib.sha256(logical_bytes(t)).hexdigest()
def pins():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
      for group in ('xiaomai_storage','xiaomai_python','tools','upstream','integration')
      for p in (ROOT/group).rglob('*.py') if '__pycache__' not in p.parts and not p.name.startswith('test_')}
def make():
    model,runtime=build(**BUILD)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    opt=PythonAdamW(named,lr=.001,betas=(.8,.95),weight_decay=.01,amsgrad=True)
    return model,runtime,opt
def evidence(runtime,*,next_step=False):
    return runtime.set_evidence(NEXT_ROWS if next_step else FIRST_ROWS,NEXT_EVENTS if next_step else FIRST_EVENTS,
                                concept_version='concept:1',source_version='synthetic-event-fixture-v090')
def objective(model,runtime,ids):
    """Same explicit multihead diagnostic objective for native and candidate."""
    with runtime.activate(training=True):
        logits=model(ids,use_cache=False).logits
        scores=runtime.score_candidates(CANDIDATES,training=True)
        tokens=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids.reshape(-1))
        action=F.cross_entropy(scores['action_logits'],torch.tensor([1],dtype=torch.long))
        outcome=F.cross_entropy(scores['outcome_logits'].reshape(-1,2),torch.tensor([0,1],dtype=torch.long))
        value=(scores['estimated_progress']-.25).square().mean()
        numeric=(scores['outcome_mean']-.5).square().mean()
        loss=tokens+.1*(action+outcome+value+numeric)
    return logits,scores,loss
def gradients(model,runtime,ids,sink=None):
    model.zero_grad(set_to_none=True)
    with PythonTape(sink=sink,record_values=True) as tape:
        with PythonResearchAdapter(runtime,tape=tape):
            logits,scores,loss=objective(model,runtime,ids)
    if loss.requires_grad or loss.grad_fn is not None:raise RuntimeError('NATIVE_AUTOGRAD_GRAPH_IN_CANDIDATE')
    tape.backward(loss);tape.assign_gradients()
    return logits,scores,loss,tape
def train(model,runtime,opt,ids,sink):
    before={n:tensor_sha(p) for n,p in model.named_parameters()}
    logits,scores,loss,tape=gradients(model,runtime,ids,sink)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    norms=clip_grad_norm_python(named,1.,sink=sink)
    old=opt.sink;opt.sink=sink
    try:opt.step()
    finally:opt.sink=old
    runtime.mark_updated()
    after={n:tensor_sha(p) for n,p in model.named_parameters()}
    changed=[n for n in after if before[n]!=after[n]]
    if not changed:raise RuntimeError('NO_RESEARCH_UPDATE')
    if any(not n.startswith('research_neural.') for n in changed):raise RuntimeError('FROZEN_BACKBONE_CHANGED')
    return {'loss_hex':float(loss).hex(),'logits_sha256':tensor_sha(logits),
      'state_sha256':tensor_sha(scores['state']),'gradient_sha256':{n:tensor_sha(p.grad) if p.grad is not None else None for n,p in named},
      'updated_parameters':changed,'clip_norm_hex':norms.hex(),'tape_nodes':len(tape.nodes),'tape_saved_elements':tape.saved,
      'runtime_version':runtime.version(),'native_backward_used':False}
def inference(model,runtime,ids):
    with torch.no_grad(),runtime.activate():
        logits=model(ids,use_cache=False).logits
        scores=runtime.score_candidates(CANDIDATES)
        return {'logits':tensor_sha(logits),'state':tensor_sha(scores['state']),
                'action_logits':tensor_sha(scores['action_logits'])}
def summary(model,runtime,opt):
    return {'parameters':{n:tensor_sha(p) for n,p in model.named_parameters()},
            'optimizer':data_hash(opt.state_dict()),'runtime_version':runtime.version(),
            'event_ledger':dict(runtime._event_tokens)}
def progress(out,name,state):
    with (out/'PROGRESS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'phase':name,'state':state,'time':time.time(),'pid':os.getpid()})+'\n');f.flush();os.fsync(f.fileno())
    print(name,state,flush=True)
def phase(out,name,fn,detail='calls',backend=True,native=False):
    progress(out,name,'BEGIN');start=time.monotonic()
    with BlockTrace(out/name,run_id='v090-research-neural',phase=name) as log:
        observer=LineTrace([ROOT/n for n in SCOPE],log.emit,opcodes=detail=='instructions') if detail!='calls' else contextlib.nullcontext()
        mathctx=candidate.PythonReferenceBackend() if backend else contextlib.nullcontext()
        guard=contextlib.nullcontext() if native else forbid_native_learning()
        with observer,guard,mathctx as bound:
            result=fn(log.emit)
            if backend:
                log.emit('BACKEND_IMPLEMENTATION_COUNTS',counts={tag:sum(x['implementation']==tag for x in bound.calls) for tag in {x['implementation'] for x in bound.calls}},
                         native_signatures=sorted({x['operator'] for x in bound.calls if x['implementation']=='DECLARED_NATIVE'}))
    trace=verify_trace(out/name,expected_seal_sha256=log.receipt['seal_sha256'])
    if detail!='calls' and trace['kinds'].get('PY_EXECUTION',0)==0:raise RuntimeError('REQUESTED_RESEARCH_SOURCE_TRACE_MISSING')
    progress(out,name,'PASS')
    return result,{'detail':detail,'scope':SCOPE,'seconds':time.monotonic()-start,'trace':trace,
                   'seal_sha256':log.receipt['seal_sha256'],'native_oracle':native,'full_model_trace':False}
def next_step(model,runtime,opt,sink):
    evidence(runtime,next_step=True);ids=torch.tensor(NEXT_IDS,dtype=torch.long)
    probe=draw()
    sink('NATIVE_RNG_PROBE',scope='test harness Python/NumPy/PyTorch RNG continuation',values=probe)
    with candidate.PythonReferenceBackend():
        out=inference(model,runtime,ids)
        updated=train(model,runtime,opt,ids,sink)
    return {'rng_probe':probe,'next_output':out,'next_update':updated,'final':summary(model,runtime,opt)}
def child(a):
    phases={}
    payload,phases['load']=phase(a.out,'load',lambda s:load_state(a.child,expected_manifest_sha256=a.manifest_sha,sink=s),a.detail,backend=False)
    if payload['source_pins']!=pins() or payload['build']!=BUILD:raise RuntimeError('CHECKPOINT_SOURCE_OR_FIXTURE_CHANGED')
    model,runtime,opt=make()
    _,phases['restore']=phase(a.out,'restore',lambda s:checkpoint.restore(model,runtime,opt,payload['state'],sink=s),a.detail,backend=False)
    result,phases['next_step']=phase(a.out,'next_step',lambda s:next_step(model,runtime,opt,s),a.detail,backend=False)
    result.update(phases=phases,pid=os.getpid())
    (a.out/'CHILD_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
def parent(a):
    begin=time.monotonic();phases={};random.seed(9028);np.random.seed(9028);torch.manual_seed(9028)
    initial_pins=pins()
    model,runtime,opt=make();evidence(runtime)
    identity={n:id(p) for n,p in model.named_parameters()}
    initial,phases['initial_capture']=phase(a.out,'initial_capture',lambda s:checkpoint.capture(model,runtime,opt,sink=s),a.detail,backend=False)
    ids=torch.tensor(FIRST_IDS,dtype=torch.long)
    initial_summary=summary(model,runtime,opt)
    def oracle(sink):
        model.zero_grad(set_to_none=True);logits,scores,loss=objective(model,runtime,ids);loss.backward()
        sink('NATIVE_ORACLE_ONLY',native_backward=True,production_research_objective=False)
        result={'logits':logits.detach().clone(),'state':scores['state'].detach().clone(),
                'loss':loss.detach().clone(),'grads':{n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}}
        return result
    ref,phases['native_oracle']=phase(a.out,'native_oracle',oracle,a.detail,backend=False,native=True)
    _,phases['restore_after_oracle']=phase(a.out,'restore_after_oracle',lambda s:checkpoint.restore(model,runtime,opt,initial,sink=s),a.detail,backend=False)
    def compare(sink):
        logits,scores,loss,tape=gradients(model,runtime,ids,sink);rows=[]
        for n,p in model.named_parameters():
            other=ref['grads'][n]
            if (p.grad is None)!=(other is None):raise RuntimeError('GRADIENT_REACHABILITY_DIFF:'+n)
            if other is not None:rows.append({'name':n,**compare_values(p.grad,other,5e-6,3e-4),'oracle_nonzero':bool(torch.any(other!=0))})
        return {'logits':compare_values(logits,ref['logits'],2e-5,2e-5),'state':compare_values(scores['state'],ref['state'],2e-5,2e-5),
                'loss':compare_values(loss,ref['loss'],2e-5,2e-5),'gradient_comparisons':rows,
                'candidate_output':tensor_sha(logits),'candidate_gradients':{n:tensor_sha(p.grad) for n,p in model.named_parameters() if p.grad is not None},
                'tape_nodes':len(tape.nodes),'tape_saved_elements':tape.saved}
    comparison,phases['python_compare']=phase(a.out,'python_compare',compare,a.detail)
    checkpoint.restore(model,runtime,opt,initial)
    with forbid_native_learning(),candidate.PythonReferenceBackend():
        control=compare(None)
    if comparison['candidate_output']!=control['candidate_output'] or comparison['candidate_gradients']!=control['candidate_gradients']:raise RuntimeError('OBSERVATION_CHANGED_RESEARCH_RESULT')
    checkpoint.restore(model,runtime,opt,initial)
    baseline,phases['before_intervention']=phase(a.out,'before_intervention',lambda s:inference(model,runtime,ids),a.detail)
    def intervention(sink):
        p=runtime.components.bridge.output.weight
        with torch.no_grad():p.zero_()
        runtime.mark_updated();changed=inference(model,runtime,ids)
        if changed['logits']==baseline['logits']:raise RuntimeError('BRIDGE_INTERVENTION_NO_EFFECT')
        checkpoint.restore(model,runtime,opt,initial,sink=sink)
        reverted=inference(model,runtime,ids)
        if reverted!=baseline:raise RuntimeError('INTERVENTION_ROLLBACK_DIFF')
        return {'changed':changed,'reverted':reverted}
    intervention_result,phases['bridge_intervention']=phase(a.out,'bridge_intervention',intervention,a.detail)
    first,phases['first_update']=phase(a.out,'first_update',lambda s:train(model,runtime,opt,ids,s),a.detail)
    if runtime._inference_key is not None or runtime._inference_state is not None:raise RuntimeError('UPDATE_DID_NOT_INVALIDATE_RESEARCH_STATE')
    _,phases['rebuild']=phase(a.out,'rebuild',lambda s:inference(model,runtime,ids),a.detail)
    state,phases['capture']=phase(a.out,'capture',lambda s:checkpoint.capture(model,runtime,opt,sink=s),a.detail,backend=False)
    payload={'build':BUILD,'source_pins':pins(),'state':state}
    saved,phases['save']=phase(a.out,'save',lambda s:save(payload,a.out/'READABLE_CHECKPOINT',sink=s),a.detail,backend=False)
    # All random probes are performed outside the candidate arithmetic backend.
    expected,phases['parent_next_step']=phase(a.out,'parent_next_step',lambda s:next_step(model,runtime,opt,s),a.detail,backend=False)
    childout=a.out/'child';childout.mkdir();cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',str(a.out/'READABLE_CHECKPOINT'),'--manifest-sha',saved['manifest_sha256'],'--out',str(childout),'--detail',a.detail]
    with (a.out/'child.stdout').open('wb') as out,(a.out/'child.stderr').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1'),stdout=out,stderr=err)
        try:code=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            import psutil
            try:
                owned=psutil.Process(proc.pid)
                for childproc in reversed(owned.children(recursive=True)):
                    try:childproc.kill()
                    except psutil.NoSuchProcess:pass
                owned.kill()
            except psutil.NoSuchProcess:pass
            proc.wait()
            raise RuntimeError('RESEARCH_CHILD_TIMEOUT')
    if code:raise RuntimeError('RESEARCH_CHILD_FAILED')
    got=json.loads((childout/'CHILD_RESULT.json').read_text(encoding='utf8'))
    if any(got[k]!=v for k,v in expected.items()):raise RuntimeError('RESEARCH_CHILD_NEXT_STEP_DIFF')
    verify_state(a.out/'READABLE_CHECKPOINT',expected_manifest_sha256=saved['manifest_sha256'])
    rolled,phases['rollback']=phase(a.out,'rollback',lambda s:checkpoint.restore(model,runtime,opt,initial,sink=s),a.detail,backend=False)
    if {n:id(p) for n,p in model.named_parameters()}!=identity:raise RuntimeError('ROLLBACK_PARAMETER_IDENTITIES_CHANGED')
    final=summary(model,runtime,opt)
    if final['parameters']!=initial_summary['parameters'] or final['optimizer']!=initial_summary['optimizer'] or final['runtime_version']!=initial_summary['runtime_version']:raise RuntimeError('ROLLBACK_RESEARCH_STATE_DIFF')
    if 'evidence:next' not in runtime._event_tokens:raise RuntimeError('ROLLBACK_ERASED_KNOWN_OBSERVATION')
    with candidate.PythonReferenceBackend():
        if inference(model,runtime,ids)!=baseline:raise RuntimeError('FINAL_ROLLBACK_OUTPUT_DIFF')
    if pins()!=initial_pins:raise RuntimeError('SOURCE_CHANGED_DURING_RESEARCH_LIFECYCLE')
    result={'status':'PINNED_RESEARCH_PYTHON_TRAINING_LITERAL_CONTINUATION_PASS','build':BUILD,'parameters':sum(p.numel() for p in model.parameters()),
      'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'comparison':comparison,'observer_on_off_output_and_gradients_bitwise_equal':True,
      'bridge_intervention_changed_output':True,'intervention_reverted_bitwise':True,'first_update':first,'checkpoint':saved,
      'new_process_next_output_update_optimizer_rng_equal':True,'child_exit_code':code,'child_pid':proc.pid,'child_command':cmd,
      'rollback_parameter_identities_preserved':True,'rollback_preserves_new_observation_ledger':True,'rollback_receipt':rolled,
      'phases':phases,'child_phases':got['phases'],'source_pins':initial_pins,'source_unchanged':True,'seconds':time.monotonic()-begin,
      'real_weights':False,'current_true':False,'GPU':False,'autonomous_research_verified':False,'production_research_objective':False,
      'entire_model_line_opcode_coverage':False,'original_research_native_source_unchanged':True}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','parameters','trainable_parameters','seconds')}))
def main():
    if sys.flags.optimize:raise RuntimeError('ASSERTIONS_REQUIRED')
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--detail',choices=('calls','lines','instructions'),default='calls');p.add_argument('--child',type=Path);p.add_argument('--manifest-sha');a=p.parse_args()
    torch.set_num_threads(1)
    if a.child:
        if not a.out.is_dir() or not a.manifest_sha:raise RuntimeError('CHILD_MANIFEST_REQUIRED')
        child(a)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        try:parent(a)
        except BaseException as exc:
            (a.out/'FAILED.json').write_text(json.dumps({'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),'pid':os.getpid()},indent=2),encoding='utf8');raise
if __name__=='__main__':main()
