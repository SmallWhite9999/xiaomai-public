"""Run the unchanged small research witness with a 60-second process limit.

This is a packaging/timeout launcher, not an implementation of an alternative
training system. The original witness performs both a separately labelled native
oracle and the guarded PythonTape/PythonAdamW candidate run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

BASE = Path(__file__).resolve().parent

def stop_tree(proc):
    try:
        owned = psutil.Process(proc.pid)
        for child in reversed(owned.children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        owned.kill()
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=10)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--detail", choices=("calls", "lines"), default="calls")
    p.add_argument("--source-root", type=Path, default=BASE.parent / "vendor/lifecycle_source")
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("OUTPUT_MUST_NOT_ALREADY_EXIST")
    source = args.source_root.resolve()
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for r in manifest["files"]:
        raw = (source / r["path"]).read_bytes()
        if len(raw) != r["bytes"] or hashlib.sha256(raw).hexdigest() != r["sha256"]:
            raise ValueError("SOURCE_HASH_MISMATCH:" + r["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    stdout = out.parent / (out.name + ".stdout.txt")
    stderr = out.parent / (out.name + ".stderr.txt")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONPATH=str(source))
    cmd = [sys.executable, "-u", "-B", str(source / "tools/run_research_lifecycle.py"),
           "--out", str(out), "--detail", args.detail]
    begin = time.monotonic()
    with stdout.open("xb") as so, stderr.open("xb") as se:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=so, stderr=se)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            stop_tree(proc)
            print(json.dumps({"status": "TIMEOUT_60_SECONDS", "exit_code": proc.returncode}))
            return 124
    elapsed = time.monotonic() - begin
    if code:
        print(json.dumps({"status": "FAILED", "exit_code": code, "seconds": elapsed}))
        return code
    original = json.loads((out / "RESULT.json").read_text(encoding="utf-8"))
    summary = {k: original[k] for k in (
        "status", "build", "parameters", "trainable_parameters", "comparison",
        "observer_on_off_output_and_gradients_bitwise_equal", "bridge_intervention_changed_output",
        "intervention_reverted_bitwise", "first_update", "new_process_next_output_update_optimizer_rng_equal",
        "child_exit_code", "rollback_parameter_identities_preserved", "rollback_preserves_new_observation_ledger",
        "source_unchanged", "real_weights", "current_true", "GPU", "autonomous_research_verified",
        "production_research_objective", "entire_model_line_opcode_coverage", "original_research_native_source_unchanged")}
    summary.update(schema="XIAOMAI_PUBLIC_SYNTHETIC_RESEARCH_LIFECYCLE_SUMMARY_V1",
                   runner_timeout_seconds=60, runner_elapsed_seconds=elapsed,
                   network_model_downloads_disabled=True, cuda_visible_devices="",
                   candidate_native_backward=False, candidate_optimizer="PythonAdamW",
                   synthetic_fixture_only=True, source_manifest_name="SOURCE_MANIFEST.json",
                   source_manifest_sha256=hashlib.sha256((source / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
                   package_inputs=[{k: r[k] for k in ("path", "bytes", "sha256")} for r in manifest["files"]],
                   checkpoint_manifest_sha256=original["checkpoint"]["manifest_sha256"],
                   tested_runtime={name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "transformers", "numba", "psutil")})
    (out / "PUBLIC_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "parameters", "trainable_parameters", "runner_elapsed_seconds")}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
