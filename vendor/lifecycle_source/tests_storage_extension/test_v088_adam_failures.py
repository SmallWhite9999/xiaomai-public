import copy
import pytest,torch
from xiaomai_storage.adam_optimizer import PythonAdam,PythonAdamW,AdamError,sha


def setup():
    a=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    b=torch.nn.Parameter(torch.tensor([3.,4.],dtype=torch.float64))
    for p in (a,b):p.grad=torch.tensor([.1,-.2],dtype=p.dtype)
    opt=PythonAdamW([('a',a),('b',b)],amsgrad=True)
    return a,b,opt

@pytest.mark.parametrize('when',['before_first','after_first','after_second','on_committed_event'])
def test_partial_commit_recovery(when):
    a,b,opt=setup();opt.step();checkpoint=opt.state_dict();original=(sha(a),sha(b));identity=(id(a),id(b))
    if when=='on_committed_event':
        def sink(kind,**fields):
            if kind=='PY_ADAM_COMMITTED':raise OSError('injected sink failure')
        opt.sink=sink
    def before():
        if when=='before_first':raise RuntimeError('injected before first')
    def after(name,p):
        if name==('a' if when=='after_first' else 'b') and when in ('after_first','after_second'):
            raise RuntimeError('injected after write')
    with pytest.raises((RuntimeError,OSError)):opt.step(before_commit=before,after_parameter_write=after)
    assert (sha(a),sha(b))==original and (id(a),id(b))==identity and opt.failed
    assert opt.step_calls==checkpoint['step_calls']
    with pytest.raises(AdamError):opt.step()
    opt.sink=None;opt.load_state_dict(checkpoint);opt.step();assert not opt.failed

@pytest.mark.parametrize('mutation',['param','grad','rebind_grad','moment','lr','param_list','negative_zero_lr'])
def test_drift_rejected_before_own_write(mutation):
    a,b,opt=setup();opt.step();before=sha(b)
    if mutation=='negative_zero_lr':opt.param_groups[0]['lr']=-0.
    def mutate():
        with torch.no_grad():
            if mutation=='param':a[0]+=1
            elif mutation=='grad':a.grad[0]+=1
            elif mutation=='rebind_grad':a.grad=a.grad.clone()
            elif mutation=='moment':opt.state['a']['exp_avg'][0]+=1
            elif mutation=='lr':opt.param_groups[0]['lr']*=2
            elif mutation=='negative_zero_lr':opt.param_groups[0]['lr']=0.
            else:opt.param_groups[0]['params'].reverse()
    with pytest.raises(AdamError):opt.step(before_commit=mutate)
    assert sha(b)==before and opt.last_receipt['writes_attempted']==0

@pytest.mark.parametrize('field',['grad','parameter','moment'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_not_committed(field,value):
    a,b,opt=setup();opt.step();before=sha(b)
    with torch.no_grad():
        if field=='grad':a.grad[0]=value
        elif field=='parameter':a[0]=value
        else:opt.state['a']['exp_avg'][0]=value
    with pytest.raises(AdamError):opt.step()
    assert sha(b)==before

@pytest.mark.parametrize('options',[{'lr':-1.},{'eps':-1.},{'betas':(1.,.9)},{'betas':(.9,-.1)},
    {'weight_decay':float('nan')},{'maximize':1},{'amsgrad':1},{'betas':[.9,.99]}])
def test_invalid_options(options):
    p=torch.nn.Parameter(torch.ones(2))
    with pytest.raises(AdamError):PythonAdam([('p',p)],**options)


def test_duplicate_storage_rejected():
    x=torch.ones(4);a=torch.nn.Parameter(x[:2]);b=torch.nn.Parameter(x[2:])
    with pytest.raises(AdamError):PythonAdam([('a',a),('b',b)])


def test_element_limit_is_precommit():
    a,b,opt=setup();opt.max_state_elements=3
    before=sha(a),sha(b)
    with pytest.raises(AdamError,match='LIMIT'):opt.step()
    assert (sha(a),sha(b))==before


def test_foreign_write_after_earlier_commit_is_detected_and_rolled_back():
    a,b,opt=setup();old=sha(a),sha(b)
    def mutate(name,p):
        if name=='b':
            with torch.no_grad():a[0]+=5
    with pytest.raises(AdamError):opt.step(after_parameter_write=mutate)
    assert (sha(a),sha(b))==old


def test_final_observer_state_mutation_is_detected():
    a,b,opt=setup();old=sha(a),sha(b)
    def sink(kind,**fields):
        if kind=='PY_ADAM_COMMITTED':opt.state['a']['exp_avg'][0]+=1
    opt.sink=sink
    with pytest.raises(AdamError):opt.step()
    assert (sha(a),sha(b))==old and opt.failed
