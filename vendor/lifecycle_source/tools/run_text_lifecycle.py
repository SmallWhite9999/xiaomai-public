"""Original text-model bodies + Python backend + literal checkpoint + phase trace.

This acceptance harness has synthetic weights/config, explicit framework/cache
adapters, and bounded greedy generation. It is not the current complete True.
All detailed events emitted in every listed phase are retained losslessly. The
chosen Python source set is explicit; native/opaque boundaries remain disclosed.
"""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.text_model_entry import TextEntry,GenerationSession,candidate,TextCache,model_digest
import tools.text_model_entry as entry
import numpy as np
import torch
from xiaomai_storage import literal_state
from xiaomai_storage.line_trace import LineTrace
from xiaomai_storage.block_trace import BlockTrace as CompressedTrace,verify as verify_compressed
from xiaomai_storage.kernels import logical_bytes
from xiaomai_observe.journal import Journal,verify as verify_journal
from tools.native_component_fixture import SOURCE

def tensor_hash(t):return hashlib.sha256(logical_bytes(t)).hexdigest()
def data_hash(v):
    if type(v) is torch.Tensor:return {'shape':list(v.shape),'dtype':str(v.dtype),'hash':tensor_hash(v)}
    if type(v) is dict or type(v).__name__=='OrderedDict':return {str(k):data_hash(x) for k,x in v.items()}
    if type(v) in (list,tuple):return [data_hash(x) for x in v]
    if type(v) is float:return {'float':v.hex()}
    return v

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'torch':torch.get_rng_state(),
            'numpy':(n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4]))}
def set_rng(r):
    random.setstate(r['python']);torch.set_rng_state(r['torch']);n=r['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
def draw():return {'python':random.random().hex(),'numpy':np.random.random(3).tobytes().hex(),'torch':tensor_hash(torch.rand(3,dtype=torch.float64))}

def selected_sources():
    files={Path(__file__).resolve(),Path(entry.__file__).resolve(),SOURCE.resolve(),Path(candidate.__file__).resolve()}
    files.update(p.resolve() for p in (ROOT/'xiaomai_storage').glob('*.py')
                 if p.name not in ('backend.py','compressed_trace.py','line_trace.py','code_identity.py','cpu_compiled.py','block_trace.py'))
    files.update(Path(m.__file__).resolve() for m in (candidate.core,candidate.training_ops,candidate.selection_ops,candidate.native_ops,candidate.streaming_mm))
    # Actual Python optimizer logic, not the C++ autograd scheduler.
    import torch.optim.sgd as sgd_module
    files.add(Path(sgd_module.__file__).resolve())
    return sorted(files)

def source_pin():
    files=selected_sources();return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'dependency:'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def progress(root,phase,status):
    with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'phase':phase,'status':status,'pid':os.getpid(),'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
    print(phase,status,flush=True)

def phase(root,name,fn,detail):
    progress(root,name,'BEGIN');start=time.perf_counter()
    journal=Journal(root/(name+'_dispatch'),run_id='v086',request_id='v086-lifecycle',phase=name,fsync_each=False)
    trace_root=root/(name+'_python')
    try:
        with CompressedTrace(trace_root,run_id='v086',phase=name) as stream:
            with LineTrace(selected_sources(),stream.emit,opcodes=detail=='instructions') as tracer:
                with candidate.PythonReferenceBackend(journal=journal,scalar_events=True,streaming_matmul=True) as backend:
                    result=fn(journal.emit)
        kinds={label:sum(row['implementation']==label for row in backend.calls) for label in {r['implementation'] for r in backend.calls}}
        if kinds.get('UNDECLARED'):raise RuntimeError('UNDECLARED_PATH')
        journal.seal(execution='RETURNED',coverage={'phase':name,'detail':detail,'explicit_python_sources':True,
           'native_dependencies':True,'opaque_locals':'marked unread','full_true':False})
        verified=verify_journal(journal.root)
        trace=verify_compressed(trace_root,expected_seal_sha256=stream.receipt['seal_sha256'])
        if trace['kinds'].get('PY_EXECUTION',0)==0:raise RuntimeError('PHASE_HAS_NO_REQUESTED_LINE_TRACE')
        receipt={'status':'PHASE_PASS','elapsed_seconds':time.perf_counter()-start,'dispatch':verified,
           'dispatch_counts':kinds,'native_signatures':sorted({r['operator'] for r in backend.calls if r['implementation']=='DECLARED_NATIVE'}),
           'python_trace':trace,'trace_seal_sha256':stream.receipt['seal_sha256'],'line_or_opcode_events':tracer.events,
           'opcode_events':tracer.opcode_events,'branches':tracer.branches}
        progress(root,name,'PASS');return result,receipt
    except BaseException:
        if not journal.closed:journal.abort()
        progress(root,name,'FAILED');raise

def optimize(model,opt,ids):
    opt.zero_grad();logits=model(ids);loss=logits.square().mean();loss.backward();opt.step()
    return {'output':tensor_hash(logits),'loss':float(loss.detach()).hex()}

def child(args):
    out=args.out
    phases={}
    def read_state(sink):
        return literal_state.load(args.child,expected_manifest_sha256=args.manifest_sha,sink=sink)
    package,phases['restore_data']=phase(out,'restore_data',read_state,args.detail)
    if package['source_pins']!=source_pin():raise RuntimeError('CHILD_SOURCE_PIN_MISMATCH')
    model=TextEntry();opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    def bind(sink):
        model.load_state_dict(package['model']);opt.load_state_dict(package['optimizer']);set_rng(package['rng'])
        return GenerationSession.from_data(model,package['session'])
    session,phases['bind_state']=phase(out,'bind_state',bind,args.detail)
    probe=draw()
    generated,phases['next_generation']=phase(out,'next_generation',lambda sink:session.feed(package['next_ids'],sink=sink),args.detail)
    result,phases['next_update']=phase(out,'next_update',lambda sink:optimize(model,opt,package['next_ids']),args.detail)
    receipt={'generated':{'logits':tensor_hash(generated[0]),'token':generated[1]},'update':result,
             'model':model_digest(model),'optimizer':data_hash(opt.state_dict()),'rng_probe':probe,
             'phases':phases,'pid':os.getpid()}
    (out/'CHILD_RESULT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')

def parent(args):
    out=args.out;started=time.time();phases={}
    random.seed(432);np.random.seed(432);torch.manual_seed(432)
    model=TextEntry();opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    ids=torch.tensor([[1,2]])
    initial={k:t.detach().clone() for k,t in model.state_dict().items()}
    initial_opt=copy.deepcopy(opt.state_dict());initial_rng=rng_state();initial_model_hash=model_digest(model)
    objects={n:id(p) for n,p in model.named_parameters()}
    with torch.no_grad():native=model(ids)
    baseline,phases['prefill']=phase(out,'prefill',lambda sink:model(ids),args.detail)
    torch.testing.assert_close(baseline,native,atol=2e-5,rtol=2e-5)
    chosen=model.language_model.layers[0].input_layernorm.weight;saved=chosen.detach().clone()
    def intervene(sink):
        with torch.no_grad():chosen.add_(.125)
        return model(ids)
    changed,phases['intervention']=phase(out,'intervention',intervene,args.detail)
    if torch.equal(changed,baseline):raise RuntimeError('INTERVENTION_HAS_NO_EFFECT')
    def restore(sink):
        with torch.no_grad():chosen.copy_(saved)
        return model(ids)
    reverted,phases['revert']=phase(out,'revert',restore,args.detail)
    if not torch.equal(reverted,baseline):raise RuntimeError('REVERT_FAILED')
    old_session=GenerationSession(model)
    _,phases['cache_before_update']=phase(out,'cache_before_update',lambda sink:old_session.feed(ids,sink=sink),args.detail)
    old_hash=model_digest(model)
    _,phases['update']=phase(out,'update',lambda sink:optimize(model,opt,ids),args.detail)
    if model_digest(model)==old_hash:raise RuntimeError('NO_UPDATE')
    def stale(sink):
        try:old_session.feed(torch.tensor([[3]]),sink=sink)
        except ValueError as exc:
            if 'STALE_CACHE' not in str(exc):raise
            sink('EXPECTED_STALE_CACHE_REJECTION',reason=str(exc));return True
        raise RuntimeError('STALE_CACHE_ACCEPTED')
    _,phases['stale_cache_check']=phase(out,'stale_cache_check',stale,args.detail)
    session=GenerationSession(model)
    generated,phases['fresh_generation']=phase(out,'fresh_generation',lambda sink:session.feed(ids,sink=sink),args.detail)
    checkpoint=out/'READABLE_CHECKPOINT'
    def save_data(sink):
        return literal_state.save({'source_pins':source_pin(),'model':model.state_dict(),'optimizer':opt.state_dict(),
             'rng':rng_state(),'session':session.data(),'next_ids':torch.tensor([[generated[1]]])},checkpoint,sink=sink)
    cp,phases['save']=phase(out,'save',save_data,args.detail)
    # Read the exact next ids from the payload the parent just committed.
    with torch.no_grad():next_ids=torch.tensor([[generated[1]]])
    probe=draw()
    nxt,phases['next_generation']=phase(out,'next_generation',lambda sink:session.feed(next_ids,sink=sink),args.detail)
    updated,phases['next_update']=phase(out,'next_update',lambda sink:optimize(model,opt,next_ids),args.detail)
    expected={'generated':{'logits':tensor_hash(nxt[0]),'token':nxt[1]},'update':updated,
              'model':model_digest(model),'optimizer':data_hash(opt.state_dict()),'rng_probe':probe}
    childout=out/'child';childout.mkdir()
    cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',str(checkpoint),'--manifest-sha',cp['manifest_sha256'],
         '--out',str(childout),'--detail',args.detail]
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    with (out/'child.stdout').open('wb') as so,(out/'child.stderr').open('wb') as se:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=so,stderr=se);code=process.wait()
    if code:raise RuntimeError('CHILD_FAILED:'+str(code))
    received=json.loads((childout/'CHILD_RESULT.json').read_text())
    if any(expected[k]!=received[k] for k in expected):raise RuntimeError('CHILD_CONTINUATION_MISMATCH')
    literal_state.verify(checkpoint,expected_manifest_sha256=cp['manifest_sha256'])
    def rollback(sink):
        model.load_state_dict(initial);opt.load_state_dict(initial_opt);set_rng(initial_rng);return model(ids)
    rolled,phases['rollback']=phase(out,'rollback',rollback,args.detail)
    if not torch.equal(rolled,baseline) or model_digest(model)!=initial_model_hash:raise RuntimeError('ROLLBACK_VALUES_FAILED')
    if data_hash(opt.state_dict())!=data_hash(initial_opt) or data_hash(rng_state())!=data_hash(initial_rng):raise RuntimeError('ROLLBACK_STATE_FAILED')
    if {n:id(p) for n,p in model.named_parameters()}!=objects:raise RuntimeError('PARAMETER_OBJECT_CHANGED')
    result={'status':'PINNED_TEXT_ENTRY_LITERAL_LIFECYCLE_PASS','parameter_count':sum(p.numel() for p in model.parameters()),
      'source_receipt':model.receipt,'parent_pid':os.getpid(),'child_pid':process.pid,'child_exit_code':code,'child_command':cmd,
      'native_python_max_abs_difference':float((native-baseline).abs().max().detach()),'tolerance':{'atol':2e-5,'rtol':2e-5},
      'intervention_changed':True,'revert_bitwise_equal':True,'updated':True,'stale_cache_rejected':True,
      'new_process_next_generation_and_update_equal':True,'optimizer_rng_equal':True,'parameter_identity_preserved':True,
      'rollback_bitwise_equal':True,'pickle_used_for_checkpoint':False,'checkpoint':cp,'detail':args.detail,
      'every_listed_phase_has_python_trace':True,'phases':phases,'child_phases':received['phases'],
      'elapsed_seconds':time.time()-started,'real_weights':False,'full_current_true':False,'gpu':False,
      'all_third_party_lines_and_gpu_internals':False,
      'saved_graph_version_counter_topology_restored':False}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','detail','parameter_count')}))

def main():
    if sys.flags.optimize:raise RuntimeError('OPTIMIZATION_NOT_ADMITTED_IN_ACCEPTANCE')
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--detail',choices=('lines','instructions'),default='lines')
    p.add_argument('--child',type=Path);p.add_argument('--manifest-sha');args=p.parse_args()
    if args.child:
        if not args.out.is_dir() or not args.manifest_sha:raise RuntimeError('CHILD_EXPLICIT_CUSTODY_REQUIRED')
        child(args)
    else:
        args.out.mkdir(parents=True,exist_ok=False);parent(args)
if __name__=='__main__':main()
