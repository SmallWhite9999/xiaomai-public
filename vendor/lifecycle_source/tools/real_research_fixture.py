"""True-size ancestor layer 0 and true embedding rows, with pinned research code.

This is explicitly one GDN layer followed by the ancestor's final norm, NOT
the original 32-layer text model or current True. No language-model head is
loaded and the returned hidden features are not token logits. Research branch
weights are explicit deterministic experimental initialization.
"""
from __future__ import annotations
import hashlib
from types import SimpleNamespace
import torch
from torch import nn
from tools.native_decoder_fixture import definitions
from tools.research_fixture import native_modules
from xiaomai_python.real_assets import AssetError,REVISION,VerifiedSnapshot
from xiaomai_python.real_layer import build_first_layer,embedding_input,EMBEDDING,PREFIX,tensor_hash

NORM='model.language_model.norm.weight'

class DeclaredEmbeddingRows(nn.Module):
    """Only explicitly declared real rows, with original vocabulary ID meaning."""
    def __init__(self,ids,values,vocabulary):
        super().__init__()
        if type(ids) is not list or not ids or len(set(ids))!=len(ids):raise ValueError('UNIQUE_DECLARED_IDS_REQUIRED')
        if any(type(n) is not int or not 0<=n<vocabulary for n in ids):raise ValueError('DECLARED_ID_OUT_OF_VOCABULARY')
        if values.ndim!=2 or values.shape[0]!=len(ids):raise ValueError('ROW_SHAPE_MISMATCH')
        self.declared_ids=tuple(ids);self.num_embeddings=vocabulary;self.embedding_dim=values.shape[1]
        self.weight=nn.Parameter(values,requires_grad=False)
    def forward(self,ids):
        if ids.dtype is not torch.long or ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('ONE_LONG_TOKEN_STREAM_REQUIRED')
        with torch._C._DisableTorchDispatch():values=ids.detach().tolist()
        lookup={n:i for i,n in enumerate(self.declared_ids)}
        if any(n not in lookup for row in values for n in row):raise ValueError('UNDECLARED_EMBEDDING_ROW_NO_SUBSTITUTION')
        indices=torch.tensor([lookup[n] for row in values for n in row],dtype=torch.long,device=self.weight.device)
        return self.weight.index_select(0,indices).reshape(*ids.shape,self.embedding_dim)

class RealLayerBackbone(nn.Module):
    def __init__(self,layer,norm,embedding,config):
        super().__init__();self.config=config
        self.embed_tokens=embedding;self.layers=nn.ModuleList([layer]);self.norm=norm
    def forward(self,input_ids=None,*,attention_mask=None,use_cache=False,return_dict=True,inputs_embeds=None):
        if use_cache is not False or return_dict is not True or inputs_embeds is not None:
            raise ValueError('REAL_LAYER_FULL_PREFIX_IDS_ONLY_NO_CACHE')
        if attention_mask is not None:
            if tuple(attention_mask.shape)!=tuple(input_ids.shape) or not bool(torch.all(attention_mask==1)):
                raise ValueError('REAL_LAYER_UNPADDED_STREAM_ONLY')
        hidden=self.embed_tokens(input_ids)
        hidden=self.layers[0](hidden,position_embeddings=None,attention_mask=None,past_key_values=None)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))

class RealLayerResearchModel(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.model=nn.Module();self.model.add_module('language_model',backbone);self.config=backbone.config
    def forward(self,input_ids=None,*,use_cache=False,attention_mask=None):
        return self.model.language_model(input_ids=input_ids,use_cache=use_cache,attention_mask=attention_mask,return_dict=True)

def _install(backbone,*,width,slots,steps,seed,nonzero_bridge):
    _,_,native=native_modules()
    model=RealLayerResearchModel(backbone);model.requires_grad_(False);model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        runtime=native.install_research_neural(model,width=width,slots=slots,steps=steps,layer=0,outcome_classes=2)
        if torch.count_nonzero(runtime.components.bridge.output.weight):raise ValueError('PINNED_ZERO_BRIDGE_CHANGED')
        if nonzero_bridge:
            with torch.no_grad():runtime.components.bridge.output.weight.copy_(torch.randn_like(runtime.components.bridge.output.weight)*.01)
            runtime.mark_updated()
    return model,runtime

def build(snapshot,*,declared_ids,width=4,slots=2,steps=1,seed=9128,nonzero_bridge=True,progress=None):
    """Return model, original research runtime, and reload/custody manifest.

    All base parameters remain their full dimensions and are frozen FP32 casts
    of admitted BF16 values. Only the unique selected embedding rows are held.
    """
    if type(snapshot) is not VerifiedSnapshot or snapshot.receipt.get('identity')!=REVISION:
        raise AssetError('PINNED_PRODUCTION_SNAPSHOT_REQUIRED')
    if type(nonzero_bridge) is not bool or type(seed) is not int:raise ValueError('EXPLICIT_RESEARCH_INITIALIZATION_REQUIRED')
    ids=list(dict.fromkeys(declared_ids))
    if not ids or any(type(n) is not int for n in ids):raise ValueError('DECLARED_IDS_REQUIRED')
    layer,layer_origin=build_first_layer(snapshot,dtype_name='float32',progress=progress)
    cfg=SimpleNamespace(**snapshot.config['text_config'])
    embeddings,embedding_origin=embedding_input(snapshot,ids,dtype_name='float32')
    row=snapshot.tensors[NORM]
    if row['dtype']!='BF16' or row['shape']!=[cfg.hidden_size]:raise AssetError('REAL_FINAL_NORM_SHAPE_OR_DTYPE')
    raw=snapshot.raw(NORM)
    norm_values=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).to(torch.float32).clone()
    ns,_=definitions()
    with torch.device('meta'):norm=ns['Qwen3_5RMSNorm'](cfg.hidden_size,eps=cfg.rms_norm_eps)
    norm.load_state_dict({'weight':norm_values},strict=True,assign=True)
    backbone=RealLayerBackbone(layer,norm,DeclaredEmbeddingRows(ids,embeddings[0],snapshot.tensors[EMBEDDING]['shape'][0]),cfg)
    model,runtime=_install(backbone,width=width,slots=slots,steps=steps,seed=seed,nonzero_bridge=nonzero_bridge)
    sources={}
    for name,p in model.named_parameters(remove_duplicate=False):
        if name.startswith('research_neural.'):continue
        if name.startswith('model.language_model.layers.0.native.'):
            key=PREFIX+name.removeprefix('model.language_model.layers.0.native.')
            sources[name]={'segments':[{'tensor':key,'start':0,'count':p.numel()}]}
        elif name=='model.language_model.norm.weight':sources[name]={'segments':[{'tensor':NORM,'start':0,'count':p.numel()}]}
        elif name=='model.language_model.embed_tokens.weight':
            sources[name]={'segments':[{'tensor':EMBEDDING,'start':n*cfg.hidden_size,'count':cfg.hidden_size} for n in ids]}
        else:raise AssetError('UNMAPPED_FROZEN_PARAMETER:'+name)
    if list(model.named_buffers()):raise AssetError('UNMAPPED_FROZEN_BUFFER')
    snapshot.recheck()
    origin={'schema':'REAL_ANCESTOR_LAYER0_RESEARCH_ASSEMBLY_V1','asset_identity':REVISION,'layer_origin':layer_origin,
      'executed_original_layer_indices':[0],'original_layer_count':snapshot.config['text_config']['num_hidden_layers'],
      'declared_embedding_ids':ids,'embedding_rows':embedding_origin,'norm':{'name':NORM,'raw_sha256':hashlib.sha256(raw).hexdigest(),'loaded_sha256':tensor_hash(norm_values)},
      'frozen_sources':sources,'research_initialization':{'seed':seed,'width':width,'slots':slots,'steps':steps,'nonzero_bridge':nonzero_bridge},
      'scope':'FULL_REAL_LAYER0_THEN_REAL_FINAL_NORM_NOT_32_LAYER_TEXT_MODEL','current_true_restored':False,
      'lm_head_loaded':False,'outputs_are_hidden_features_not_logits':True,'full_embedding_loaded':False,'GPU':False}
    return model,runtime,origin
