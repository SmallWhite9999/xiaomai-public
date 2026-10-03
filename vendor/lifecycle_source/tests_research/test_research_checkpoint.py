import json,os,random,shutil,subprocess,sys,types
from pathlib import Path
import numpy as np
import pytest,torch
from tools.research_fixture import build
from xiaomai_storage.adam_optimizer import PythonAdamW
from xiaomai_storage.research_checkpoint import capture,restore,content_signature,ResearchCheckpointError
from xiaomai_storage.data_snapshot import clone_data
from xiaomai_storage.literal_state import save,load

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def objects():
    model,runtime=build()
    optimizer=PythonAdamW([(n,p) for n,p in model.named_parameters() if p.requires_grad],record_values=False)
    runtime.set_evidence([[1,2]],['old'],concept_version='c1',source_version='s1')
    runtime.set_evidence([[3,4]],['current'],concept_version='c2',source_version='s2')
    return model,runtime,optimizer

def draw():
    return random.random(),float(np.random.random()),torch.rand(3).tolist()

def advance(model,runtime,optimizer):
    name,p=next(iter(optimizer.named.items()))
    p.grad=torch.full_like(p,.125)
    optimizer.step();runtime.mark_updated()

def test_readable_roundtrip_preserves_full_history_and_parameters(objects,tmp_path):
    m,r,o=objects;advance(m,r,o)
    before_ids={n:id(p) for n,p in m.named_parameters()}
    data=capture(m,r,o);receipt=save(data,tmp_path/'saved')
    loaded=load(tmp_path/'saved',expected_manifest_sha256=receipt['manifest_sha256'])
    expected_rng=draw();advance(m,r,o)
    r.set_evidence([[5]],['new'],concept_version='c3',source_version='s3')
    result=restore(m,r,o,loaded)
    assert result['retained_live_events']==1
    assert r._snapshot==data['runtime']['snapshot']
    assert r._event_tokens=={'old':[1,2],'current':[3,4],'new':[5]}
    assert r.weight_revision==data['runtime']['weight_revision']
    assert before_ids=={n:id(p) for n,p in m.named_parameters()}
    assert draw()==expected_rng
    assert o.step_calls==data['optimizer']['step_calls']
    assert set(o.state)==set(data['optimizer']['state'])
    for n,t in data['parameters'].items():assert torch.equal(dict(m.named_parameters())[n],t)
    with pytest.raises(ValueError,match='immutable'):r.set_evidence([[9]],['old'])

@pytest.mark.parametrize('relocate',[False,True])
def test_second_process_restores_different_hashseed(objects,tmp_path,relocate):
    m,r,o=objects;advance(m,r,o);data=capture(m,r,o)
    saved=save(data,tmp_path/'saved');expected=content_signature(data)
    code='''import json,sys
from tools.research_fixture import build
from xiaomai_storage.adam_optimizer import PythonAdamW
from xiaomai_storage.literal_state import load
from xiaomai_storage.research_checkpoint import restore,capture,content_signature
m,r=build();o=PythonAdamW([(n,p) for n,p in m.named_parameters() if p.requires_grad],record_values=False)
d=load(sys.argv[1],expected_manifest_sha256=sys.argv[2]);receipt=restore(m,r,o,d)
print(json.dumps({'receipt':receipt,'signature':content_signature(capture(m,r,o))}))
'''
    child_root=ROOT
    if relocate:
        child_root=tmp_path/'new_extraction'
        shutil.copytree(ROOT,child_root,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    env=dict(os.environ,PYTHONPATH=str(child_root),PYTHONHASHSEED='777',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    p=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path/'saved'),saved['manifest_sha256']],cwd=child_root,env=env,capture_output=True,timeout=90)
    assert p.returncode==0,p.stderr.decode(errors='replace')
    row=json.loads(p.stdout)
    assert row['signature']==[expected[0],list(expected[1])]
    assert row['receipt']['historical_events']==2
    assert row['receipt']['retained_live_events']==0

@pytest.mark.parametrize('context',('stack','encode','lease','adapter'))
@pytest.mark.parametrize('operation',('capture','restore'))
def test_active_runtime_rejected(objects,context,operation):
    m,r,o=objects;data=capture(m,r,o)
    if context=='stack':r.binding.stack.append(object())
    elif context=='encode':r.binding.encode_depth=1
    elif context=='lease':m._ornith_research_cache_lease=object()
    else:r._python_research_adapter=object()
    with pytest.raises(ResearchCheckpointError):
        capture(m,r,o) if operation=='capture' else restore(m,r,o,data)

@pytest.mark.parametrize('field',('sources','configuration','topology','lease','missing_parameter','shape','gradient_alias','ledger','counter','rng','optimizer'))
def test_bad_saved_payload_rejected_before_model_mutation(objects,field):
    m,r,o=objects;data=capture(m,r,o);before=content_signature(data);name=next(iter(data['parameters']))
    if field=='sources':data['sources']['base_commit']='0'*40
    elif field=='configuration':data['configuration']['width']+=1
    elif field=='topology':data['topology']['parameters'][name]['offset']+=1
    elif field=='lease':data['ephemeral']['decode_lease']=True
    elif field=='missing_parameter':data['parameters'].pop(name)
    elif field=='shape':data['parameters'][name]=torch.zeros(1)
    elif field=='gradient_alias':data['gradients'][name]=data['parameters'][name]
    elif field=='ledger':data['runtime']['event_tokens'].pop('current')
    elif field=='counter':data['runtime']['weight_revision']=-1
    elif field=='rng':data['rng']['python']=(99,(),None)
    else:data['optimizer']['step_calls']=-1
    with pytest.raises((ValueError,RuntimeError)):restore(m,r,o,data)
    assert content_signature(capture(m,r,o))==before

def test_historical_conflict_rejected_even_outside_frontier(objects):
    m,r,o=objects;data=capture(m,r,o);data['runtime']['event_tokens']['old']=[9]
    before=content_signature(capture(m,r,o))
    with pytest.raises(ResearchCheckpointError,match='IMMUTABLE_EVENT_CONTENT_CONFLICT'):restore(m,r,o,data)
    assert content_signature(capture(m,r,o))==before

def test_latent_cache_is_discarded_not_trusted(objects):
    m,r,o=objects;r._inference_state=torch.tensor([float('nan')]);r._inference_key=('obsolete',object())
    data=capture(m,r,o)
    assert '_inference_state' not in data['runtime'] and '_inference_key' not in data['runtime']
    restore(m,r,o,data)
    assert r._inference_state is None and r._inference_key is None

def test_plain_checkpoint_does_not_alias_live_history_or_storage(objects):
    m,r,o=objects;data=capture(m,r,o);name=next(iter(data['parameters']))
    original=data['parameters'][name].clone()
    with torch.no_grad():dict(m.named_parameters())[name].add_(1)
    r._event_tokens['old'].append(8)
    assert torch.equal(data['parameters'][name],original)
    assert data['runtime']['event_tokens']['old']==[1,2]

def test_restored_runtime_does_not_alias_caller_payload(objects):
    m,r,o=objects;data=capture(m,r,o);restore(m,r,o,data)
    data['runtime']['event_tokens']['old'].append(8)
    data['runtime']['snapshot']['token_rows'][0].append(9)
    assert r._event_tokens['old']==[1,2] and r._snapshot['token_rows']==[[3,4]]

def test_capture_observer_change_cannot_return_success(objects):
    m,r,o=objects
    def observer(kind,**kwargs):
        if kind=='RESEARCH_CHECKPOINT_CAPTURED':r.weight_revision+=1
    with pytest.raises(ResearchCheckpointError,match='CHANGED_DURING_CAPTURE'):capture(m,r,o,sink=observer)

@pytest.mark.parametrize('phase',('RESEARCH_CHECKPOINT_RESTORE_PREPARED','RESEARCH_CHECKPOINT_RESTORED'))
def test_restore_observer_mutation_rejected_and_rolled_back(objects,phase):
    m,r,o=objects;data=capture(m,r,o);advance(m,r,o);before=content_signature(capture(m,r,o))
    def observer(kind,**kwargs):
        if kind==phase:r.weight_revision+=3
    with pytest.raises(ResearchCheckpointError):restore(m,r,o,data,sink=observer)
    assert content_signature(capture(m,r,o))==before

def test_failed_optimizer_requires_explicit_restore(objects):
    m,r,o=objects;data=capture(m,r,o);o.failed=True
    with pytest.raises(ResearchCheckpointError,match='FAILED_OPTIMIZER'):capture(m,r,o)
    restore(m,r,o,data);assert not o.failed

def test_model_optimizer_parameter_replacement_rejected(objects):
    m,r,o=objects;name=next(iter(o.named));p=o.named[name]
    parent=m
    for segment in name.split('.')[:-1]:parent=getattr(parent,segment)
    setattr(parent,name.split('.')[-1],torch.nn.Parameter(p.detach().clone()))
    with pytest.raises(ResearchCheckpointError,match='BINDING_CHANGED'):capture(m,r,o)

def test_registered_backward_hook_not_silently_omitted(objects):
    m,r,o=objects;p=next(iter(o.named.values()));handle=p.register_hook(lambda grad:grad)
    try:
        with pytest.raises(ResearchCheckpointError,match='HOOK'):capture(m,r,o)
    finally:handle.remove()

def test_configuration_mismatch_in_new_fixture_rejected(objects):
    m,r,o=objects;data=capture(m,r,o)
    other,rr=build(width=6);oo=PythonAdamW([(n,p) for n,p in other.named_parameters() if p.requires_grad])
    with pytest.raises(ResearchCheckpointError,match='CONFIGURATION'):restore(other,rr,oo,data)


@pytest.mark.parametrize('mutation',('root_instance','nested_instance','class','same_code_instance','defaults','wrapper_inner_defaults'))
def test_loaded_forward_drift_rejected_before_any_restore_write(objects,monkeypatch,mutation):
    m,r,o=objects;data=capture(m,r,o)
    advance(m,r,o)
    before=content_signature(capture(m,r,o))
    def replaced(self,*args,**kwargs):return None
    # setitem restores absence of the instance entry. setattr's generic undo
    # would leave a bound method override for the formerly inherited method.
    if mutation=='root_instance':monkeypatch.setitem(vars(m),'forward',types.MethodType(replaced,m))
    elif mutation=='nested_instance':
        nested=m.model.language_model.layers[0].mlp
        monkeypatch.setitem(vars(nested),'forward',types.MethodType(replaced,nested))
    elif mutation=='class':monkeypatch.setattr(type(m),'forward',replaced)
    elif mutation=='same_code_instance':
        monkeypatch.setitem(vars(m),'forward',types.MethodType(type(m).forward,m))
    elif mutation=='defaults':monkeypatch.setattr(type(m).forward,'__defaults__',(False,))
    else:
        outer=m.model.language_model.rotary_emb.forward.__func__
        inner=dict(zip(outer.__code__.co_freevars,(c.cell_contents for c in outer.__closure__)))['func']
        monkeypatch.setattr(inner,'__defaults__',(42,))
    with pytest.raises(ResearchCheckpointError,match='CONFIGURATION_MISMATCH'):
        restore(m,r,o,data)
    monkeypatch.undo()
    assert content_signature(capture(m,r,o))==before
