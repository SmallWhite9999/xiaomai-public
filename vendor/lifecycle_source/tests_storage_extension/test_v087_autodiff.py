import math
import pytest,torch
from xiaomai_storage.python_autodiff import PythonTape,TapeError
from tools.integration_backend import load
C=load()

OPS={
 'square':lambda x:x.square().sum(),
 'exp':lambda x:x.exp().sum(),
 'log':lambda x:x.log().sum(),
 'rsqrt':lambda x:x.rsqrt().sum(),
 'sqrt':lambda x:x.sqrt().sum(),
 'sin':lambda x:x.sin().sum(),
 'cos':lambda x:x.cos().sum(),
 'tanh':lambda x:x.tanh().sum(),
 'sigmoid':lambda x:x.sigmoid().sum(),
 'silu':lambda x:torch.nn.functional.silu(x).sum(),
 'softplus':lambda x:torch.nn.functional.softplus(x,beta=1.5,threshold=5).sum(),
 'pow':lambda x:x.pow(3).sum(),
 'slice':lambda x:x.t()[:,::2].square().sum(),
 'split':lambda x:sum(t.square().sum() for t in x.split(1)),
 'cumsum':lambda x:x.cumsum(-1).square().sum(),
 'pad':lambda x:torch.nn.functional.pad(x,(1,2,-1,0)).square().sum(),
 'triu':lambda x:x.triu(1).sum(),
 'tril':lambda x:x.tril(-1).sum(),
 'softmax':lambda x:(x.softmax(-1)*x).sum(),
 'logsoftmax':lambda x:x.log_softmax(-1).sum(),
 'broadcast':lambda x:(x*x.mean(0,keepdim=True)+x.sum(1,keepdim=True)).mean(),
 'cat':lambda x:torch.cat([x,x[:,1:]],1).square().mean(),
 'masked':lambda x:x.masked_fill(torch.tensor([[False,True,False],[False,False,True]]),0).sum(),
 'clonecast':lambda x:x.float().clone().double().sum(),
 'detach':lambda x:(x*x.detach()).sum(),
}
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('name',list(OPS))
def test_vjp_against_native_engine(name,dtype):
 torch.manual_seed(510)
 x=(torch.rand(2,3,dtype=dtype)+.5).requires_grad_()
 with C.PythonReferenceBackend():
  loss=OPS[name](x);reference=torch.autograd.grad(loss,x)[0]
 with C.PythonReferenceBackend(),PythonTape() as tape:
  result=OPS[name](x)
 assert not result.requires_grad and result.grad_fn is None
 tape.backward(result)
 torch.testing.assert_close(tape.gradient(x),reference,atol=3e-6,rtol=3e-5)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('batch',[False,True])
def test_multiplication_two_independent_leaves(dtype,batch):
 torch.manual_seed(515)
 a=torch.randn((2,3,4) if batch else (3,4),dtype=dtype,requires_grad=True)
 b=torch.randn((2,4,5) if batch else (4,5),dtype=dtype,requires_grad=True)
 with C.PythonReferenceBackend():
  loss=(a@b).square().sum();ga,gb=torch.autograd.grad(loss,(a,b))
 with C.PythonReferenceBackend(),PythonTape() as tape:loss=(a@b).square().sum()
 tape.backward(loss)
 torch.testing.assert_close(tape.gradient(a),ga,atol=2e-5,rtol=5e-5)
 torch.testing.assert_close(tape.gradient(b),gb,atol=2e-5,rtol=5e-5)

def test_indexed_write_through_view_routes_gradient():
 x=torch.tensor([2.,3.,4.],dtype=torch.float64,requires_grad=True)
 def forward():
  out=torch.zeros(5,dtype=x.dtype)
  out[1:4].copy_(x*x)
  out[2:3].copy_(x[:1]+1)
  return out.sum()
 with C.PythonReferenceBackend():ref=torch.autograd.grad(forward(),x)[0]
 with C.PythonReferenceBackend(),PythonTape() as tape:y=forward()
 tape.backward(y)
 assert torch.equal(tape.gradient(x),ref)
 assert any(n.op=='copy_storage_update' and n.active for n in tape.nodes)

def test_overwrite_cuts_previous_assignment_gradient():
 x=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
 with C.PythonReferenceBackend(),PythonTape() as tape:
  out=torch.zeros_like(x);out.copy_(x);out.copy_(torch.ones_like(x));loss=out.sum()
 tape.backward(loss)
 assert torch.equal(tape.gradient(x),torch.zeros_like(x))

def test_all_native_backward_entrypoints_forbidden(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('native autograd was called')
 monkeypatch.setattr(torch.autograd,'backward',forbidden)
 monkeypatch.setattr(torch.autograd,'grad',forbidden)
 monkeypatch.setattr(torch.Tensor,'backward',forbidden)
 x=torch.tensor([2.,3.],requires_grad=True)
 with C.PythonReferenceBackend(),PythonTape() as tape:y=(x*x).sum()
 tape.backward(y);tape.assign_gradients()
 assert torch.equal(x.grad,torch.tensor([4.,6.]))

def test_unrecorded_raw_mutation_rejected():
 x=torch.tensor([2.,3.],requires_grad=True)
 with PythonTape() as t:y=(x*x).sum()
 x.data[0]=9
 with pytest.raises(TapeError,match='LEAF_VALUES'):t.backward(y)

def test_mid_forward_external_write_detected():
 x=torch.tensor([2.,3.],requires_grad=True)
 with pytest.raises(TapeError,match='UNRECORDED'),PythonTape():
  y=x*x
  with torch._C._DisableTorchDispatch():x.data[0]=7
  y+x

def test_mutating_external_leaf_rejected_before_write():
 x=torch.tensor([2.,3.],requires_grad=True);old=x.detach().clone()
 with pytest.raises(TapeError,match='EXTERNAL_SOURCE'),PythonTape():x.copy_(torch.ones_like(x))
 assert torch.equal(x,old)

def test_resource_bound_rejects_and_restores_grad_mode():
 x=torch.ones(10,requires_grad=True)
 with pytest.raises(TapeError,match='RESOURCE'),PythonTape(max_saved_elements=5):x*x
 assert torch.is_grad_enabled()

def test_unknown_active_operation_is_not_silently_detached():
 x=torch.randn(4,requires_grad=True)
 with pytest.raises(TapeError,match='VJP_NOT'),PythonTape():x.relu()

def test_independent_leaf_aliases_not_misidentified():
 x=torch.nn.Parameter(torch.ones(4));y=torch.nn.Parameter(x.detach())
 with pytest.raises(TapeError,match='SHARE_STORAGE'),PythonTape():x+y

def test_explicit_external_nonleaf_rejected():
 x=torch.ones(4,requires_grad=True);y=x*2
 with pytest.raises(TapeError,match='NONLEAF'),PythonTape():y*y

def test_nonfinite_gradient_logged_not_zeroed():
 events=[];x=torch.tensor([0.],requires_grad=True)
 with PythonTape(sink=lambda kind,**kw:events.append((kind,kw))) as t:y=x.log().sum()
 with pytest.raises(TapeError,match='NONFINITE'):t.backward(y)
 assert t.failed and not t.backward_done
 assert events[-1][0]=='PY_BACKWARD_FAILED'

def test_native_grad_cannot_be_reenabled_inside_tape():
 x=torch.ones(2,requires_grad=True)
 with pytest.raises(TapeError,match='REENABLED'),PythonTape(),torch.enable_grad():x*x

def test_each_tape_used_once():
 x=torch.ones(2,requires_grad=True)
 with PythonTape() as t:y=x.sum()
 t.backward(y)
 with pytest.raises(TapeError):t.backward(y)

def test_reverse_step_data_are_available_to_observer():
 events=[];x=torch.tensor([2.,3.],requires_grad=True)
 with PythonTape(sink=lambda kind,**kw:events.append((kind,kw)),record_values=True) as t:y=x.square().sum()
 t.backward(y)
 assert any(k=='PY_TAPE_NODE' and 'values' in v for k,v in events)
 assert any(k=='PY_BACKWARD_NODE' and 'gradient' in v for k,v in events)
 assert events[-1][0]=='PY_BACKWARD_END'

@pytest.mark.parametrize('seed',[4,17,29])
def test_finite_difference_independent_numeric_oracle(seed):
 torch.manual_seed(seed)
 x=(torch.rand(2,3,dtype=torch.float64)+.25).requires_grad_()
 def f(z):return (z.sin()*z.sigmoid()+z.pow(3)).mean()
 with C.PythonReferenceBackend(),PythonTape() as t:y=f(x)
 t.backward(y);g=t.gradient(x);eps=1e-6
 for i in range(x.numel()):
  a=x.detach().clone();b=x.detach().clone();a.reshape(-1)[i]+=eps;b.reshape(-1)[i]-=eps
  fd=(float(f(a))-float(f(b)))/(2*eps)
  assert abs(float(g.reshape(-1)[i])-fd)<1e-8


def test_tape_enter_logger_error_restores_grad_state():
 def broken(*a,**k):raise OSError('broken')
 before=torch.is_grad_enabled()
 with pytest.raises(OSError),PythonTape(sink=broken):pass
 assert torch.is_grad_enabled()==before


def test_non_differentiable_output_not_claimed_as_success():
 x=torch.tensor([1.,2.],requires_grad=True)
 with PythonTape() as tape:y=x.detach().sum()
 with pytest.raises(TapeError,match='NO_DIFFERENTIABLE'):tape.backward(y)
 assert not tape.backward_done


def test_storage_capacity_guard_happens_before_values_snapshot(monkeypatch):
 x=torch.ones(100,requires_grad=True)
 tape=PythonTape(max_saved_elements=5)
 def forbidden(*a,**k):raise AssertionError('read values before validating budget')
 monkeypatch.setattr(tape,'values',forbidden)
 with pytest.raises(TapeError,match='RESOURCE'),tape:x*x
