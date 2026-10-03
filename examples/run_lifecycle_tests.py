"""Run original gradient, interrupted-update and restore rejection tests."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from run_lifecycle_demo import stop_tree

BASE = Path(__file__).resolve().parent
FILES = ("tests_storage_extension/test_v087_autodiff.py",
         "tests_storage_extension/test_v088_adam_failures.py",
         "tests_research/test_research_checkpoint.py")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=BASE.parent / "vendor/lifecycle_source")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source_root.resolve()
    cmd = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *FILES,
           "--junitxml=" + str(out / "tests.xml")]
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    begin = time.monotonic()
    with (out / "stdout.txt").open("xb") as so, (out / "stderr.txt").open("xb") as se:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=so, stderr=se)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            stop_tree(proc)
            print(json.dumps({"status": "TEST_TIMEOUT_60_SECONDS", "exit_code": proc.returncode}))
            return 124
    suites = ET.parse(out / "tests.xml").getroot()
    summary = {"schema": "XIAOMAI_PUBLIC_ORIGINAL_LIFECYCLE_TESTS_V1", "status": "PASS" if code == 0 else "FAILED",
               "exit_code": code, "timeout_seconds": 60, "seconds": time.monotonic() - begin,
               "test_files": list(FILES), "tests": sum(int(s.get("tests", "0")) for s in suites.findall("testsuite")),
               "failures": sum(int(s.get("failures", "0")) for s in suites.findall("testsuite")),
               "errors": sum(int(s.get("errors", "0")) for s in suites.findall("testsuite")),
               "skipped": sum(int(s.get("skipped", "0")) for s in suites.findall("testsuite")),
               "synthetic_cpu_only": True, "private_true_checkpoint_read": False}
    (out / "PUBLIC_TEST_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
