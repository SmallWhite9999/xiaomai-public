"""Exact pinned native definitions with SMALL synthetic fixtures, not True 9B.

Only the named top-level definitions are compiled from the verified source AST.
Locations and bodies are unchanged. Dependencies/configs are explicitly supplied
for component engineering checks, not substituted into the production loader.
"""
from __future__ import annotations
import __future__
import ast
import hashlib
from pathlib import Path
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE=Path(__file__).resolve().parents[1]/'upstream/modeling_qwen3_5.py'
SHA256='395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6'
NAMES=('Qwen3_5RMSNormGated','Qwen3_5RMSNorm','Qwen3_5MLP',
       'Qwen3_5VisionMLP','Qwen3_5VisionPatchEmbed','Qwen3_5VisionPatchMerger',
       'torch_causal_conv1d_update','apply_mask_to_padding_states',
       'repeat_kv','eager_attention_forward')

def load_definitions():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256:raise ValueError('PINNED_NATIVE_SOURCE_MISMATCH')
    tree=ast.parse(raw,filename=str(SOURCE));nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in NAMES]
    if {n.name for n in nodes}!=set(NAMES):raise ValueError('NATIVE_DEFINITION_MISSING')
    namespace={'__name__':'xiaomai_cpu_native_fixture','torch':torch,'nn':nn,'F':F,
       'ACT2FN':{'silu':F.silu,'gelu':F.gelu,'gelu_pytorch_tanh':lambda x:F.gelu(x,approximate='tanh')}}
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(module,str(SOURCE),'exec',flags=__future__.annotations.compiler_flag),namespace)
    receipt={'source_file':str(SOURCE),'sha256':SHA256,
       'definitions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,
                       'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()} for n in nodes],
       'scope':'EXACT_NATIVE_COMPONENT_BODIES_SYNTHETIC_CONFIG_AND_INPUT_NOT_TRUE',
       'real_true_loaded':False}
    return namespace,receipt

def build_component(kind,dtype=torch.float64):
    ns,receipt=load_definitions()
    if kind=='text_mlp':
        model=ns['Qwen3_5MLP'](types.SimpleNamespace(hidden_size=4,hidden_act='silu'),7)
        inputs=(torch.randn(1,3,4,dtype=dtype),)
    elif kind=='vision_mlp':
        model=ns['Qwen3_5VisionMLP'](types.SimpleNamespace(hidden_size=4,intermediate_size=7,hidden_act='gelu_pytorch_tanh'))
        inputs=(torch.randn(4,4,dtype=dtype),)
    elif kind=='vision_embed':
        model=ns['Qwen3_5VisionPatchEmbed'](types.SimpleNamespace(hidden_size=4,patch_size=2,temporal_patch_size=2,in_channels=1))
        inputs=(torch.randn(2,8,dtype=dtype),)
    elif kind.startswith('vision_merger'):
        model=ns['Qwen3_5VisionPatchMerger'](types.SimpleNamespace(hidden_size=2,spatial_merge_size=2,out_hidden_size=3),use_postshuffle_norm=kind.endswith('post'))
        inputs=(torch.randn(8,2,dtype=dtype),)
    elif kind=='rms':
        model=ns['Qwen3_5RMSNorm'](4)
        inputs=(torch.randn(2,3,4,dtype=dtype),)
    elif kind=='rms_gated':
        model=ns['Qwen3_5RMSNormGated'](4)
        inputs=(torch.randn(2,3,4,dtype=dtype),torch.randn(2,3,4,dtype=dtype))
    else:raise ValueError(kind)
    return model.to(dtype=dtype),inputs,receipt
