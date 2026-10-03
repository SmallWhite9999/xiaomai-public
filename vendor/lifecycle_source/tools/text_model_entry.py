"""Pinned Qwen3.5 TextModel/Rotary bodies driven through the Python backend.

This advances the integration boundary beyond the hand-written DecoderStack.
The complete text-model and rotary class bodies are compiled unedited from the
pinned file. Framework decorators, pretrained base initialization and the cache/
mask adapters below remain explicitly narrowed laboratory scaffolding. This is
NOT a full transformers installation or current True restoration.
"""
from __future__ import annotations
import ast,contextlib,hashlib,types,typing
from pathlib import Path
import torch
from torch import nn
from tools.integration_backend import load
candidate=load()
from tools.native_decoder_fixture import definitions,config,set_chunk_size,ReferenceCache
from tools.native_component_fixture import SOURCE,SHA256
from xiaomai_storage import numeric_words as words,kernels as storage

class TextCache(ReferenceCache):
    def __init__(self,config=None,layers=None):
        if config is None and layers is None:raise ValueError('CACHE_LAYER_COUNT_REQUIRED')
        self.layer_types=None if config is None else tuple(config.layer_types)
        super().__init__(len(config.layer_types) if config is not None else layers)
    def get_seq_length(self,*args):
        for layer in self.layers:
            if layer.keys is not None:return layer.keys.shape[-2]
        return 0
    def has_previous_state(self,index=None):
        if index is not None:return super().has_previous_state(index)
        return any(layer.recurrent_states is not None for layer in self.layers)
    @classmethod
    def from_data(cls,data,config):
        if len(data['layers'])!=len(config.layer_types):raise ValueError('CACHE_LAYER_COUNT_MISMATCH')
        result=cls(config=config)
        for layer,values in zip(result.layers,data['layers']):layer.__dict__.update(values)
        result.seen=data['seen'];return result

class FrameworkBase(nn.Module):
    def __init__(self,config):super().__init__();self.config=config
    def post_init(self):pass  # random nn initialization is fixture data, not pretrained weights

def identity_decorator(fn=None,**kwargs):
    return (lambda f:f) if fn is None else fn

def mask_adapter(*,config,inputs_embeds,attention_mask,past_key_values,position_ids):
    b,t,d=inputs_embeds.shape
    if t<1:raise ValueError('NONEMPTY_TEXT_REQUIRED')
    seen=0 if past_key_values is None else past_key_values.get_seq_length()
    length=seen+t
    if position_ids is not None:
        if tuple(position_ids.shape)!=(b,t):raise ValueError('TEXT_POSITION_SHAPE')
        # This adapter handles ordinary un-packed sequential text only.
        expected=torch.arange(seen,length,device=inputs_embeds.device).expand(b,t)
        if not torch.equal(position_ids,expected):raise ValueError('PACKED_OR_CUSTOM_POSITIONS_NOT_ADMITTED')
    if attention_mask is not None:
        if inputs_embeds.dtype is torch.float64 and not bool(torch.all(attention_mask==1)):
            raise ValueError('PADDED_FP64_MASK_ADAPTER_NOT_ADMITTED')
        if tuple(attention_mask.shape)!=(b,length):raise ValueError('MASK_PREFIX_SHAPE')
        if not bool(torch.all(torch.logical_or(attention_mask==0,attention_mask==1))):raise ValueError('BINARY_MASK_REQUIRED')
    # Python constructs the causal selection geometry. Tensor/scalar operations
    # below go through the active candidate dispatcher, not a vendor mask kernel.
    shape=(b,1,t,length)
    result=torch.zeros(shape,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
    for batch in range(b):
        for q in range(t):
            for key in range(length):
                visible=key<=seen+q and (attention_mask is None or bool(attention_mask[batch,key]))
                if not visible:result[batch,0,q,key]=torch.finfo(inputs_embeds.dtype).min
    return result

def text_definitions():
    ns,receipt=definitions()
    ns.update({'Qwen3_5PreTrainedModel':FrameworkBase,'Qwen3_5TextConfig':types.SimpleNamespace,
               'Callable':typing.Callable,'Optional':typing.Optional,
               'ROPE_INIT_FUNCTIONS':{},'dynamic_rope_update':identity_decorator,
               'maybe_autocast':lambda **kwargs:contextlib.nullcontext(),
               'merge_with_config_defaults':identity_decorator,'capture_outputs':identity_decorator,
               'auto_docstring':identity_decorator,'create_causal_mask':mask_adapter,
               'DynamicCache':TextCache,'Qwen3_5ModelOutputWithPast':lambda **k:types.SimpleNamespace(**k),
               'BaseModelOutputWithPast':types.SimpleNamespace})
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA256:raise ValueError('SOURCE_CHANGED')
    names={'Qwen3_5TextRotaryEmbedding','Qwen3_5TextModel'}
    nodes=[n for n in ast.parse(raw).body if type(n) is ast.ClassDef and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('TEXT_CLASS_SET')
    full=compile(raw,str(SOURCE),'exec',dont_inherit=True)
    code={c.co_name:c for c in full.co_consts if type(c) is types.CodeType and c.co_name in names}
    selected=compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec',dont_inherit=True)
    constants=tuple(code[c.co_name] if type(c) is types.CodeType and c.co_name in names else c for c in selected.co_consts)
    exec(selected.replace(co_consts=constants),ns)
    receipt={**receipt,'text_entry_class_bodies_unmodified':True,
             'additional_definitions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,
                'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()} for n in nodes],
             'remaining_scaffolding':['identity framework decorators','nn.Module pretrained base without post_init work',
                'plain output container','text-only sequential causal mask adapter','reference hybrid cache',
                'default RoPE only, autocast disabled','synthetic reduced config and random weights']}
    return ns,receipt

class TextEntry(nn.Module):
    def __init__(self,layer_types=('linear_attention','linear_attention','linear_attention','full_attention'),*,dtype=torch.float64,vocab_size=17):
        if type(vocab_size) is not int or vocab_size<1:raise ValueError('VOCAB_SIZE_REQUIRED')
        super().__init__();self.config=config(layer_types)
        self.config.vocab_size=vocab_size;self.config.pad_token_id=None
        self.config.num_hidden_layers=len(layer_types);self.config.max_position_embeddings=128
        self.config.rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[1,1,0]}
        self.config.use_cache=True
        if not any(t=='full_attention' for t in layer_types):raise ValueError('FIXTURE_REQUIRES_FULL_ATTENTION_FOR_CACHE_LENGTH')
        ns,self.receipt=text_definitions()
        self.language_model=ns['Qwen3_5TextModel'](self.config)
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        self.to(dtype=dtype);self.eval();set_chunk_size(self,2)
    def forward(self,input_ids=None,*,cache=None,attention_mask=None,inputs_embeds=None):
        # Pass use_cache explicitly because the framework default merger is an
        # admitted identity decorator in this diagnostic assembly.
        result=self.language_model(input_ids=input_ids,inputs_embeds=inputs_embeds,
             attention_mask=attention_mask,past_key_values=cache,use_cache=cache is not None)
        return self.lm_head(result.last_hidden_state)
    def new_cache(self):return TextCache(config=self.config)

def model_digest(model):
    h=hashlib.sha256()
    import json
    h.update(json.dumps(vars(model.config),sort_keys=True,separators=(',',':')).encode())
    combined=list(model.state_dict().items())+[('BUFFER:'+n,t) for n,t in model.named_buffers()]
    for name,t in combined:
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        # Bounded byte chunks; synthetic witness weights are small. True 9B
        # throughput and persistent cache signatures are separate acceptance.
        source=storage.raw(t)
        for start in range(0,len(source),65536):h.update(source[start:start+65536])
    return h.hexdigest()

def greedy_ids(logits,*,sink=None):
    if logits.ndim!=3 or logits.shape[1]<1:raise ValueError('LOGITS_BTV')
    storage.admit(logits);out=[];buf=storage.raw(logits);width=logits.element_size()
    for batch in range(logits.shape[0]):
        winner=None;best=None
        for token in range(logits.shape[2]):
            offset=storage.offset(logits,(batch,logits.shape[1]-1,token))*width
            bits=bytes(buf[offset:offset+width]);value=words.decode(bits,logits.dtype)
            if value!=value:raise ValueError('NAN_LOGIT_NOT_REPAIRED')
            choose=winner is None or value>best
            if sink is not None:sink('GREEDY_COMPARE',batch=batch,token=token,value_hex=value.hex(),
                                     value_bits=bits.hex(),previous_winner=winner,selected=choose)
            if choose:winner=token;best=value
        out.append(winner)
    return out

class GenerationSession:
    """Bounded greedy-generation component; not the full True agent/tool harness."""
    def __init__(self,model):
        self.model=model;self.cache=model.new_cache();self.parameter_digest=model_digest(model);self.prefix=[];self.failed=False
    def feed(self,ids,*,sink=None):
        if self.failed:raise ValueError('GENERATION_SESSION_FAILED_REBUILD_REQUIRED')
        if model_digest(self.model)!=self.parameter_digest:raise ValueError('STALE_CACHE_AFTER_PARAMETER_CHANGE')
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:raise ValueError('SINGLE_BATCH_NONEMPTY_TOKEN_IDS_REQUIRED')
        storage.admit(ids)
        if ids.dtype is not torch.int64 and ids.dtype is not torch.int32:raise ValueError('INTEGER_TOKEN_IDS_REQUIRED')
        if self.cache.get_seq_length()!=len(self.prefix):raise ValueError('CACHE_PREFIX_LENGTH_MISMATCH')
        buf=storage.raw(ids);width=ids.element_size();items=[]
        for j in range(ids.shape[1]):
            at=storage.offset(ids,(0,j))*width;ident=words.decode(bytes(buf[at:at+width]),ids.dtype)
            if not 0<=ident<self.model.config.vocab_size:raise ValueError('TOKEN_OUT_OF_VOCABULARY')
            items.append(ident)
        try:
            with torch.no_grad():logits=self.model(ids,cache=self.cache)
            winner=greedy_ids(logits,sink=sink)[0]
            self.prefix.extend(items)
            if self.cache.get_seq_length()!=len(self.prefix):raise ValueError('CACHE_PREFIX_LENGTH_MISMATCH')
            if sink is not None:sink('GENERATION_FORWARD',new_input_tokens=ids.shape[1],seen=self.cache.get_seq_length(),model_sha256=self.parameter_digest)
            return logits,winner
        except BaseException:
            self.failed=True;raise
    def data(self):
        if self.failed:raise ValueError('GENERATION_SESSION_FAILED_REBUILD_REQUIRED')
        if model_digest(self.model)!=self.parameter_digest:raise ValueError('STALE_CACHE_AFTER_PARAMETER_CHANGE')
        if self.cache.get_seq_length()!=len(self.prefix):raise ValueError('CACHE_PREFIX_LENGTH_MISMATCH')
        return {'parameter_digest':self.parameter_digest,'prefix':self.prefix.copy(),'cache':self.cache.data()}
    @classmethod
    def from_data(cls,model,data):
        if model_digest(model)!=data['parameter_digest']:raise ValueError('RESTORED_CACHE_WEIGHT_MISMATCH')
        out=cls(model);out.cache=TextCache.from_data(data['cache'],model.config);out.prefix=list(data['prefix'])
        if out.cache.get_seq_length()!=len(out.prefix):raise ValueError('CACHE_PREFIX_LENGTH_MISMATCH')
        return out
