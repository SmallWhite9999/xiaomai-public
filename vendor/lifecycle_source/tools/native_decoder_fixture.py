"""Full pinned decoder bodies + explicitly synthetic integration scaffolding.

No pretrained weights or True graph are loaded. The arithmetic class definitions
are compiled unedited from the admitted source. The cache, stack assembly,
configuration, positions and base-class scaffold below are reference harnesses,
not claims of a complete Hugging Face/True deployment.
"""
from __future__ import annotations
import __future__
import ast
import functools
import hashlib
from pathlib import Path
import types
import typing
import torch
from torch import nn
import torch.nn.functional as F
from tools.native_component_fixture import SOURCE,SHA256
from xiaomai_python.training_ops import compose_attention_mask
from xiaomai_python.dispatch import array,tensor

NAMES=(
 'Qwen3_5RMSNormGated','apply_mask_to_padding_states','torch_causal_conv1d_update',
 'l2norm','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
 'Qwen3_5GatedDeltaNet','rotate_half','apply_rotary_pos_emb','repeat_kv',
 'eager_attention_forward','Qwen3_5Attention','Qwen3_5MLP','Qwen3_5RMSNorm',
 'Qwen3_5DecoderLayer',
)

class EagerOnly:
    def get_interface(self,name,default):
        if name!='eager':raise ValueError('FIXTURE_EAGER_ONLY')
        return default

class QuietLogger:
    def warning_once(self,*args):pass


def definitions():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256:raise ValueError('PINNED_SOURCE_MISMATCH')
    tree=ast.parse(raw,filename=str(SOURCE))
    nodes=[n for n in tree.body if type(n) in (ast.ClassDef,ast.FunctionDef) and n.name in NAMES]
    if {n.name for n in nodes}!=set(NAMES):raise ValueError('NATIVE_CLASS_SET_MISMATCH')
    ns={'__name__':'xiaomai_pinned_decoder_fixture','torch':torch,'nn':nn,'F':F,
        'ACT2FN':{'silu':F.silu},'GradientCheckpointingLayer':nn.Module,
        'Qwen3_5Config':types.SimpleNamespace,'Qwen3_5TextConfig':types.SimpleNamespace,
        'Cache':ReferenceCache,'Unpack':typing.Unpack,
        'FlashAttentionKwargs':typing.TypedDict('FixtureFlashAttentionKwargs',{}),
        'TransformersKwargs':typing.TypedDict('FixtureTransformersKwargs',{}),
        'FusedRMSNormGated':None,'causal_conv1d_fn':None,'causal_conv1d_update':None,
        'chunk_gated_delta_rule':None,'fused_recurrent_gated_delta_rule':None,
        'is_fast_path_available':False,'logger':QuietLogger(),'ALL_ATTENTION_FUNCTIONS':EagerOnly()}
    # Compiling a subset alone can change CPython's imported-module call
    # optimization. Keep the definition code objects compiled in FULL source
    # context, while executing only the explicitly selected definitions.
    full=compile(raw,str(SOURCE),'exec',dont_inherit=True)
    definitions_by_key={(c.co_qualname,c.co_firstlineno):c for c in full.co_consts if type(c) is types.CodeType}
    selected=compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec',dont_inherit=True)
    constants=[]
    for c in selected.co_consts:
        if type(c) is types.CodeType:
            key=(c.co_qualname,c.co_firstlineno)
            if key not in definitions_by_key:raise ValueError('FULL_COMPILE_DEFINITION_NOT_FOUND')
            c=definitions_by_key[key]
        constants.append(c)
    exec(selected.replace(co_consts=tuple(constants)),ns)
    receipt=dict(source_sha256=SHA256,exact_definition_bodies=True,full_source_compiled_definition_code_retained=True,
       definitions=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,
         ast_sha256=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()) for n in nodes],
       scaffolding=['nn.Module instead of GradientCheckpointingLayer; no activation checkpointing',
                    'reference cache and position/mask assembly',
                    'eager-only attention dispatch', 'synthetic config and weights; annotation-only type placeholders'],
       real_true_loaded=False,checkpoint_model_weights_used=False)
    return ns,receipt

class ReferenceCache:
    def __init__(self,layers):
        self.layers=[types.SimpleNamespace(conv_states=None,recurrent_states=None,keys=None,values=None) for _ in range(layers)]
        self.seen=0
    def has_previous_state(self,index):return self.layers[index].recurrent_states is not None
    def update_conv_state(self,state,index):self.layers[index].conv_states=state;return state
    def update_recurrent_state(self,state,index):self.layers[index].recurrent_states=state;return state
    def update(self,keys,values,index):
        layer=self.layers[index]
        if layer.keys is not None:
            keys=torch.cat((layer.keys,keys),dim=-2);values=torch.cat((layer.values,values),dim=-2)
        layer.keys=keys;layer.values=values
        return keys,values
    def data(self):
        return dict(seen=self.seen,layers=[dict(vars(layer)) for layer in self.layers])
    @classmethod
    def from_data(cls,data):
        result=cls(len(data['layers']));result.seen=data['seen']
        for layer,values in zip(result.layers,data['layers']):layer.__dict__.update(values)
        return result


def config(layer_types=('linear_attention','linear_attention','linear_attention','full_attention')):
    return types.SimpleNamespace(hidden_size=4,intermediate_size=7,hidden_act='silu',
       linear_num_value_heads=2,linear_num_key_heads=1,linear_key_head_dim=2,
       linear_value_head_dim=2,linear_conv_kernel_dim=4,rms_norm_eps=1e-6,
       num_attention_heads=2,num_key_value_heads=1,head_dim=2,attention_dropout=0.,
       attention_bias=False,layer_types=list(layer_types),dtype=None,_attn_implementation='eager')


def set_chunk_size(model,size):
    if size is None:return
    if type(size) is not int or size<=0:raise ValueError('POSITIVE_CHUNK_SIZE')
    for layer in model.modules():
        if type(layer).__name__=='Qwen3_5GatedDeltaNet':
            # Same original algorithm, explicit public kernel scheduling argument.
            original=layer.chunk_gated_delta_rule
            if type(original) is functools.partial:original=original.func
            layer.chunk_gated_delta_rule=functools.partial(original,chunk_size=size)

class DecoderStack(nn.Module):
    def __init__(self,layer_types=('linear_attention','linear_attention','linear_attention','full_attention'),*,chunk_size=2,dtype=torch.float64):
        super().__init__();self.config=config(layer_types);ns,self.receipt=definitions()
        self.embed_tokens=nn.Embedding(17,4)
        self.layers=nn.ModuleList([ns['Qwen3_5DecoderLayer'](self.config,i) for i in range(len(layer_types))])
        self.norm=ns['Qwen3_5RMSNorm'](4,eps=self.config.rms_norm_eps)
        self.lm_head=nn.Linear(4,17,bias=False)
        self.to(dtype=dtype);self.eval();set_chunk_size(self,chunk_size)
        self.receipt['chunk_size']=chunk_size if chunk_size is not None else 64
    def forward(self,input_ids,cache=None):
        if input_ids.ndim!=2:raise ValueError('IDS_2D')
        hidden=self.embed_tokens(input_ids);length=hidden.shape[1];seen=0 if cache is None else cache.seen
        positions=torch.arange(seen,seen+length,dtype=torch.float32,device=hidden.device)
        # Small fixture has full rotary head_dim=2, so its sole inverse frequency is 1.
        angles=positions.reshape(1,length,1).expand(hidden.shape[0],length,2)
        cos=angles.cos().to(hidden.dtype);sin=angles.sin().to(hidden.dtype)
        mask=compose_attention_mask(length,seen+length,causal=True,query_offset=seen)
        additive=tensor(mask,dtype=torch.bool)
        attention_mask=torch.zeros_like(additive,dtype=hidden.dtype).masked_fill(~additive,-float('inf'))
        for layer in self.layers:
            hidden=layer(hidden,position_embeddings=(cos,sin),attention_mask=None if layer.layer_type=='linear_attention' else attention_mask,past_key_values=cache)
        if cache is not None:cache.seen+=length
        return self.lm_head(self.norm(hidden))
