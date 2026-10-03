"""Completion is a linked protocol, not two filenames or exit code zero.
Checks recorded evidence consistency; not hardware attestation of model math.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .long_paths import no_links, asset_child, write_json_new
from .real_assets import json_data, REVISION, PINNED_FILES
from .evidence_digest import generated_evidence_digest
from .progress import verify_progress

class ReceiptError(ValueError): pass

REQUIRED_PHASES=('source_admission','layer_assembly','embedding','native','python','python_trace',
    'identity','zero_mixer','reverted','update','updated','save','child_resume','final_rollback','source_postcheck')
SOURCE_FOLDERS=('xiaomai_python','xiaomai_observe','xiaomai_causal','tools','upstream','adapters')
COMPARISON_CONTRACT='DURABLE_COMPARISONS_V083'
TENSOR_COMPARISONS={
    'native_python':('native','python','allclose'),
    'trace_baseline':('python_trace','python','bitwise_equal'),
    'identity_baseline':('identity','python','bitwise_equal'),
    'intervention_effect':('zero_mixer','python','bitwise_different'),
    'intervention_rollback':('reverted','python','bitwise_equal'),
    'updated_output':('updated','python','bitwise_different'),
    'final_rollback':('final_rollback','python','bitwise_equal'),
}


def validate_comparisons(root,spec,result,rows,*,expected_spec_sha256,tick=None):
    read=lambda path:read_object(path,tick=tick)
    contract=spec.get('comparison_contract')
    if contract is None:
        if result.get('comparison_contract') is not None:raise ReceiptError('COMPARISON_CONTRACT_NOT_PINNED')
        return {'contract':None,'legacy_receipt':True}
    if contract!=COMPARISON_CONTRACT or result.get('comparison_contract')!=contract:
        raise ReceiptError('COMPARISON_CONTRACT_MISMATCH')
    summaries=result.get('comparisons')
    required=set(TENSOR_COMPARISONS)|{'parameter_update','non_target_parameters','resumed_parameters',
        'resumed_output','next_output','continuation_next_output_sha256','continuation_next_parameters_sha256',
        'continuation_next_optimizer_sha256','continuation_next_rng_sha256','continuation_rng_probe',
        'rollback_parameters','parameter_objects'}
    if type(summaries) is not dict or set(summaries)!=required:
        raise ReceiptError('COMPARISON_RECEIPT_SET_MISMATCH')
    for name,row in summaries.items():
        if tick:tick()
        if read(root/('COMPARISON_'+name+'.json'))!=row:
            raise ReceiptError('COMPARISON_SUMMARY_CONTRADICTION:'+name)
        if row.get('name')!=name or row.get('run_spec_sha256')!=expected_spec_sha256 or row.get('passed') is not True:
            raise ReceiptError('COMPARISON_FAILED_OR_WRONG_RUN:'+name)
        seconds=row.get('comparison_seconds')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ReceiptError('COMPARISON_TIME_INVALID:'+name)
        if row.get('current_true_restored') is not False:raise ReceiptError('COMPARISON_SCOPE_EXCEEDED')
        if row.get('schema')=='xiaomai.layer-comparison.v083':
            if row.get('same_shape') is not True or row.get('same_dtype') is not True or row.get('nonfinite_pairs')!=0:
                raise ReceiptError('COMPARISON_VALUES_INVALID:'+name)
            error=row.get('max_abs')
            if type(error) not in (int,float) or not math.isfinite(error) or error<0:
                raise ReceiptError('COMPARISON_NUMERIC_ERROR_INVALID:'+name)
            if row.get('values_sampled') is not False:raise ReceiptError('COMPARISON_SAMPLING_NOT_ADMITTED')
            relation=row.get('relation')
            if relation=='allclose':
                if row.get('elements_outside_tolerance')!=0 or row.get('max_tolerance_excess')!=0:
                    raise ReceiptError('COMPARISON_TOLERANCE_FAILED:'+name)
            elif relation in ('bitwise_equal','bitwise_different'):
                equal=row.get('actual_sha256')==row.get('reference_sha256')
                if equal!=(relation=='bitwise_equal') or row.get('bitwise_equal') is not equal:
                    raise ReceiptError('COMPARISON_BITS_CONTRADICT_RELATION:'+name)
            else:raise ReceiptError('COMPARISON_RELATION_INVALID')
        elif row.get('schema')=='xiaomai.identity-comparison.v083':
            if row.get('relation')!='data_identity_equal' or row.get('actual_sha256')!=row.get('reference_sha256'):
                raise ReceiptError('STATE_COMPARISON_CONTRADICTION:'+name)
        else:raise ReceiptError('COMPARISON_SCHEMA_INVALID:'+name)
    for name,(actual,reference,relation) in TENSOR_COMPARISONS.items():
        row=summaries[name]
        if row.get('relation')!=relation or row.get('actual_sha256')!=rows[actual]['output_sha256'] or row.get('reference_sha256')!=rows[reference]['output_sha256']:
            raise ReceiptError('COMPARISON_PHASE_LINK_MISMATCH:'+name)
    numeric=summaries['native_python']
    if numeric.get('atol')!=spec['atol'] or numeric.get('rtol')!=spec['rtol'] or numeric['max_abs']!=result['native_python_max_abs']:
        raise ReceiptError('COMPARISON_NUMERIC_SUMMARY_MISMATCH')
    restored=read(root/'child_evidence/COMPARISON_restored_saved_output.json')
    if restored.get('passed') is not True or restored.get('run_spec_sha256')!=expected_spec_sha256 or restored.get('actual_sha256')!=rows['updated']['output_sha256'] or restored.get('reference_sha256')!=rows['updated']['output_sha256']:
        raise ReceiptError('CHILD_RESTORED_COMPARISON_MISMATCH')
    for name in ('CHECKPOINT_PLAN.json','CHECKPOINT_MEASUREMENT.json'):read(root/name)
    return {'contract':contract,'comparisons_verified':len(summaries)+1,'legacy_receipt':False}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def source_inventory(root, *, tick=None):
    root=Path(root);files={}
    for folder in SOURCE_FOLDERS:
        for p in sorted((root/folder).rglob('*.py')):
            if tick:tick()
            no_links(p)
            files[p.relative_to(root).as_posix()]=sha(_read_bytes(p,tick=tick))
    return files


def verify_code(root, expected, *, tick=None):
    if type(expected) is not dict or not expected or source_inventory(root,tick=tick)!=expected:
        raise ReceiptError('CANDIDATE_SOURCE_CHANGED')


def _read_bytes(path, *, tick=None):
    pieces=[]
    if tick:tick()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8<<20):
            pieces.append(block)
            if tick:tick()
    raw=b''.join(pieces)
    if tick:tick()
    return raw


def read_object(path, *, maximum=64<<20, tick=None):
    p=no_links(path)
    if p.stat().st_size>maximum:raise ReceiptError('RECEIPT_TOO_LARGE')
    obj=json_data(_read_bytes(p,tick=tick))
    if tick:tick()
    if type(obj) is not dict:raise ReceiptError('RECEIPT_OBJECT_REQUIRED')
    return obj


def content_identity(files):
    clean={n:{'sha256':v['sha256'],'bytes':v['bytes']} for n,v in files.items()}
    return sha(json.dumps(clean,sort_keys=True,separators=(',',':')).encode())


def validate_asset_rows(files):
    if type(files) is not dict or set(files)!=set(PINNED_FILES):raise ReceiptError('SOURCE_PIN_SET_MISMATCH')
    for name,pin in PINNED_FILES.items():
        row=files[name]
        if type(row) is not dict or row.get('sha256')!=pin['sha256'] or type(row.get('bytes')) is not int or row['bytes']<=0:
            raise ReceiptError('SOURCE_PIN_MISMATCH:'+name)
        if pin['bytes'] is not None and row['bytes']!=pin['bytes']:raise ReceiptError('SOURCE_SIZE_MISMATCH:'+name)
    return content_identity(files)


def seal_evidence(root, spec_sha256, *, tick=None, progress=None):
    root=Path(root)
    files={};paths=[];total_bytes=0
    for p in root.rglob('*'):
        if tick:tick()
        no_links(p)
        paths.append(p)
    for p in sorted(paths):
        if tick:tick()
        no_links(p)
        if p.is_file():
            name=p.relative_to(root).as_posix()
            files[name]=generated_evidence_digest(p,tick=tick)
            total_bytes+=files[name]['bytes']
            if progress:progress(stage='seal_generated_evidence',file=name,
                                 files_verified=len(files),bytes_verified=total_bytes)
    manifest={'schema':'xiaomai.layer-evidence.v082','run_spec_sha256':spec_sha256,'files':files}
    if tick:tick()
    write_json_new(root/'EVIDENCE_FILES.json',manifest)
    if tick:tick()
    return manifest


def validate_completion(root, spec_path, *, expected_spec_sha256, code_root=None, worker_pids=None,tick=None,progress=None):
    read=lambda path:read_object(path,tick=tick)
    if tick:tick()
    root=no_links(root)
    raw=_read_bytes(no_links(spec_path),tick=tick)
    if sha(raw)!=expected_spec_sha256:raise ReceiptError('RUN_SPEC_CHANGED')
    spec=json_data(raw)
    if spec.get('schema')!='xiaomai.real-layer-spec.v082' or spec.get('revision')!=REVISION:
        raise ReceiptError('RUN_SPEC_SCHEMA_OR_REVISION')
    if code_root is not None:verify_code(code_root,spec['source_inventory'],tick=tick)
    if (root.parent/'WORKER_FAILURE.json').exists():raise ReceiptError('WORKER_REPORTED_FAILURE')
    result=read(root/'RESULT.json')
    source=read(root/'SOURCE_RECHECK.json')
    origin=read(root/'ORIGIN.json')
    rollback=read(root/'ROLLBACK.json')
    for obj, schema, status in ((result,'xiaomai.layer-result.v082','LAYER_DIAGNOSTIC_PASS'),
                                (source,'xiaomai.source-recheck.v082','UNCHANGED_FULL_HASH_SET')):
        if obj.get('schema')!=schema or obj.get('status')!=status:raise ReceiptError('INVALID_RECEIPT_STATUS_OR_SCHEMA')
        if obj.get('run_spec_sha256')!=expected_spec_sha256 or obj.get('revision')!=REVISION:
            raise ReceiptError('RECEIPT_SPEC_OR_REVISION_MISMATCH')
    if result.get('real_assets') is not True or result.get('asset_identity')!=REVISION:
        raise ReceiptError('NOT_REAL_ANCESTOR_ASSETS')
    if worker_pids is not None and result.get('worker_pid') not in worker_pids:
        raise ReceiptError('WORKER_IDENTITY_MISMATCH')
    for field in ('all_layer_weights_loaded','trace_output_bitwise_equal','identity_equal','intervention_effect',
                  'intervention_restored','update_effect','save_reload_new_process','parameter_identity_preserved','rollback_equal'):
        if result.get(field) is not True:raise ReceiptError('REQUIRED_CHECK_MISSING_OR_FAILED:'+field)
    for field in ('current_true_restored','gpu_run'):
        if result.get(field) is not False:raise ReceiptError('SCOPE_EXCEEDED:'+field)
    if result.get('atol')!=spec['atol'] or result.get('rtol')!=spec['rtol']:
        raise ReceiptError('TOLERANCE_CHANGED')
    if result.get('update_parameter')!=spec['update_parameter']:raise ReceiptError('UPDATE_PARAMETER_CHANGED')
    if not any(type(result.get('native_python_max_abs')) is t for t in (int,float)) or not math.isfinite(result['native_python_max_abs']) or result['native_python_max_abs']<0:
        raise ReceiptError('INVALID_NUMERIC_ERROR')
    for k in ('parameters_restored','parameter_identity_preserved','rng_restored'):
        if rollback.get(k) is not True:raise ReceiptError('ROLLBACK_NOT_CONFIRMED')
    if origin.get('run_spec_sha256')!=expected_spec_sha256 or origin.get('asset_identity')!=REVISION:
        raise ReceiptError('ORIGIN_IDENTITY_MISMATCH')
    if origin.get('parameter_count')!=218407104 or origin.get('hidden_size')!=4096 or len(origin.get('tensors',[]))!=14:
        raise ReceiptError('NOT_FULL_SIZE_LAYER0')
    if origin.get('loaded_dtype')!=spec['dtype']:raise ReceiptError('DTYPE_CHANGED')
    expected_source=spec['source_inventory'].get('upstream/modeling_qwen3_5.py')
    if not expected_source or origin.get('source_sha256')!=expected_source:raise ReceiptError('NATIVE_SOURCE_MISMATCH')
    before=validate_asset_rows(origin['custody']['files'])
    after=validate_asset_rows(source['files'])
    if before!=after or source.get('content_identity')!=before:raise ReceiptError('SOURCE_BEFORE_AFTER_CONTRADICTION')
    progress_summary=verify_progress(root/'PROGRESS.jsonl',spec_sha256=expected_spec_sha256,required_phases=REQUIRED_PHASES,tick=tick)
    manifest=read(root/'EVIDENCE_FILES.json')
    if manifest.get('schema')!='xiaomai.layer-evidence.v082' or manifest.get('run_spec_sha256')!=expected_spec_sha256:
        raise ReceiptError('EVIDENCE_MANIFEST_MISMATCH')
    files=manifest.get('files')
    required={'RESULT.json','SOURCE_RECHECK.json','ORIGIN.json','ROLLBACK.json','PROGRESS.jsonl',
              'resumed.json','saved_layer/STATE.json','saved_layer/STATE.sha256'}
    if type(files) is not dict or not required<=set(files):raise ReceiptError('EVIDENCE_FILES_MISSING')
    actual=set()
    for p in root.rglob('*'):
        if tick:tick()
        no_links(p)
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    if actual!={*files,'EVIDENCE_FILES.json'}:raise ReceiptError('EVIDENCE_FILE_SET_CHANGED')
    verified_files=0;verified_bytes=0
    for name,row in files.items():
        if tick:tick()
        generated_evidence_digest(asset_child(root,name),expected_sha256=row['sha256'],expected_bytes=row['bytes'],tick=tick)
        verified_files+=1;verified_bytes+=row['bytes']
        if progress:progress(stage='validate_generated_evidence',file=name,files_verified=verified_files,bytes_verified=verified_bytes)
    resumed=read(root/'resumed.json')
    state_sha=sha(_read_bytes(root/'saved_layer/STATE.json',tick=tick))
    if result['save']['graph_sha256']!=state_sha or resumed.get('state_sha256')!=state_sha:
        raise ReceiptError('RELOAD_STATE_IDENTITY_MISMATCH')
    if resumed.get('schema')!='xiaomai.layer-reload.v082':raise ReceiptError('RELOAD_SCHEMA_INVALID')
    if resumed.get('status')!='LAYER_RELOAD_PASS' or resumed.get('run_spec_sha256')!=expected_spec_sha256:
        raise ReceiptError('RELOAD_FAILED_OR_WRONG_RUN')
    runs=result.get('runs')
    if type(runs) is not list or not runs:raise ReceiptError('PHASE_RESULTS_LIST_REQUIRED')
    rows={}
    for row in runs:
        if type(row) is not dict or type(row.get('phase')) is not str or not row['phase']:
            raise ReceiptError('INVALID_PHASE_RESULT')
        name=row['phase']
        if name in rows:raise ReceiptError('DUPLICATE_PHASE_RESULTS')
        digest=row.get('output_sha256')
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ReceiptError('INVALID_PHASE_OUTPUT_SHA256:'+name)
        rows[name]=row
    if result.get('trace_lifecycle_enabled') is not True:raise ReceiptError('LIFECYCLE_TRACE_NOT_ENABLED')
    for name in ('native','python','python_trace','identity','zero_mixer','reverted','updated','final_rollback'):
        row=rows.get(name)
        if not row or (name!='native' and (row.get('python_backend_enabled') is not True or
                type(row.get('backend_calls')) is not list or not row['backend_calls'])):
            raise ReceiptError('PHASE_OR_DISPATCH_MISSING:'+name)
    comparisons=validate_comparisons(root,spec,result,rows,expected_spec_sha256=expected_spec_sha256,tick=tick)
    # Check the recorded outcomes as well as the worker's success flags. These
    # exact-hash relations are part of the diagnostic contract; native/Python
    # numerical equivalence remains the worker's tolerance-based comparison.
    baseline=rows['python']['output_sha256']
    for name in ('python_trace','identity','reverted','final_rollback'):
        if rows[name]['output_sha256']!=baseline:
            raise ReceiptError('PHASE_OUTPUT_EQUALITY_CONTRADICTION:'+name)
    for name in ('zero_mixer','updated'):
        if rows[name]['output_sha256']==baseline:
            raise ReceiptError('PHASE_OUTPUT_EFFECT_CONTRADICTION:'+name)
    if resumed.get('output_sha256')!=rows['updated']['output_sha256'] or resumed.get('parameters_sha256')!=result.get('updated_parameter_hashes'):
        raise ReceiptError('RELOAD_RESULT_CONTRADICTION')
    for flag in ('next_input_distinct','next_input_bitwise_equal','next_update_parameters_equal','optimizer_equal','rng_probe_equal'):
        if result.get(flag) is not True:raise ReceiptError('CONTINUATION_MISSING:'+flag)
    for field in ('next_output_sha256','next_optimizer_sha256','next_rng_sha256','rng_probe'):
        if not result.get('continuation_result') or result['continuation_result'].get(field)!=resumed.get(field):
            raise ReceiptError('CONTINUATION_RESULT_CONTRADICTION:'+field)
    if not resumed.get('next_parameters_sha256') or resumed.get('next_parameters_sha256')!=result.get('next_parameter_hashes'):
        raise ReceiptError('NEXT_UPDATE_CONTRADICTION')
    # Numerical checks remain the worker's measured assertions; trace event
    # seals and stored result hashes are independently checked here.
    from xiaomai_observe.journal import verify
    required_traces=('python_trace','identity','zero_mixer','reverted','updated','final_rollback',
                     'update_trace','save_trace','next_input_trace','next_update_trace',
                     'child_evidence/resume_forward_trace','child_evidence/next_input_trace','child_evidence/next_update_trace')
    for name in required_traces:
        if tick:tick()
        if progress:progress(stage='validate_trace',trace=name)
        trace=verify(root/name,tick=tick,file_digest=generated_evidence_digest)
        seal=read(root/name/'SEAL.json')
        if seal.get('execution')!='RETURNED':raise ReceiptError('TRACE_EXECUTION_NOT_RETURNED:'+name)
        if name in rows and rows[name].get('trace_seal')!=seal:raise ReceiptError('TRACE_SEAL_RESULT_MISMATCH:'+name)
    graph=read(root/'saved_layer/STATE.json')
    def blobs(value):
        if tick:tick()
        if type(value) is dict:
            if set(('sha256','bytes')) <= value.keys():
                rel='saved_layer/blobs/'+value['sha256']
                if rel not in files or files[rel]['sha256']!=value['sha256'] or files[rel]['bytes']!=value['bytes']:
                    raise ReceiptError('STATE_REFERENCED_BLOB_MISSING_OR_CHANGED')
            for child in value.values():blobs(child)
        elif type(value) is list:
            for child in value:blobs(child)
    blobs(graph)
    if tick:tick()
    return {'status':'EVIDENCE_CONSISTENCY_VERIFIED','run_spec_sha256':expected_spec_sha256,
            'source_content_identity':before,'progress':progress_summary,'files_verified':len(files),
            'comparisons':comparisons,'mathematical_reexecution_by_parent':False}
