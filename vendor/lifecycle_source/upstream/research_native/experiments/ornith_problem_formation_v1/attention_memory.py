"""Query-row tiling of pinned eager attention during inference, with all keys.

No history truncation, windowed attention or token budget. Per-row softmax still
sees every original key and the original additive mask. The dense attention
matrix is not returned; requests for weights/gradients use the original path.
"""
import contextlib


def query_tiled_attention(original, module, query, key, value, attention_mask,
                          scaling, dropout=0., *, tile_rows=128, **kwargs):
    import torch
    if (torch.is_grad_enabled() or module.training or kwargs.get('output_attentions') or
            query.shape[-2] <= tile_rows):
        return original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
    pieces=[]
    for start in range(0,query.shape[-2],tile_rows):
        end=min(start+tile_rows,query.shape[-2])
        mask=attention_mask
        if mask is not None and mask.shape[-2] != 1:
            mask=mask[...,start:end,:]
        output,weights=original(module,query[...,start:end,:],key,value,mask,
                                scaling=scaling,dropout=dropout,**kwargs)
        pieces.append(output)
        del weights
    return torch.cat(pieces,dim=1),None


@contextlib.contextmanager
def bounded_attention_workspace(model,trace=None):
    """Scoped to this model's Qwen text attention modules; restored on all exits.

The run has a single model execution thread. Do not overlap this context with
another generation in the same process. Training outside generate is unchanged.
"""
    modules={id(m) for m in model.modules() if type(m).__name__=='Qwen3_5Attention'} if hasattr(model,'modules') else set()
    if not modules:
        yield
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    original=implementation.eager_attention_forward
    def attention(module,*args,**kwargs):
        if id(module) in modules:
            return query_tiled_attention(original,module,*args,**kwargs)
        return original(module,*args,**kwargs)
    implementation.eager_attention_forward=attention
    try:
        if trace:
            trace.emit('ATTENTION_WORKSPACE',query_tile_rows=128,all_keys_retained=True,
                       history_truncated=False,modules=len(modules),scope='INFERENCE_ONLY')
        yield
    finally:
        implementation.eager_attention_forward=original


def probe_long_prefix(model,tokenizer,length,trace):
    """One explicit engineering test, never inserted into research history.

The single output token is a test fixture, not a cap on the research loop.
Failures are recorded and research still starts; no admission gate is added.
"""
    import torch
    import gc
    from experiments.ornith_intrinsic_branch_v1.scoped_generation_cache import generation_cache_scope
    receipt={'scope':'AUTHOR_MEMORY_PROBE_NOT_RESEARCH','input_tokens':length,'output_tokens_requested':1}
    ids=None;output=None
    try:
        ids=torch.full((1,length),tokenizer.encode('a',add_special_tokens=False)[0],
                       dtype=torch.long,device=model.get_input_embeddings().weight.device)
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        with trace.stage('long_prefix_memory_probe'),torch.inference_mode(),generation_cache_scope(model),bounded_attention_workspace(model,trace):
            output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,
                do_sample=False,max_new_tokens=1,cache_implementation='dynamic')
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
        receipt.update(status='RETURNED',actual_input_tokens=ids.shape[1],
                       actual_output_tokens=output.shape[1]-ids.shape[1],
                       peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    except Exception as exc:
        receipt.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
    finally:
        del output,ids
        gc.collect()
        torch.cuda.empty_cache()
    trace.emit('MEMORY_PROBE_RESULT',**receipt)
    return receipt
