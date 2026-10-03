"""Synthetic/public full chunk graph witness; never opens True/private state."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from xiaomai_trace.chunk_trace import CHUNK_SOURCE, NAMES, chunk_vjp, run_trace, validate_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(813)
    def draw(shape, scale):
        return (rng.normal(size=shape) * scale).astype(np.float32)
    q, k = draw((1, 3, 1, 2), .2), draw((1, 3, 1, 2), .2)
    v = draw((1, 3, 1, 2), .3)
    g = -rng.uniform(.1, .5, (1, 3, 1)).astype(np.float32)
    beta = rng.uniform(.1, .8, g.shape).astype(np.float32)
    initial, dy = draw((1, 1, 2, 2), .1), draw(v.shape, .2)
    ds = draw(initial.shape, .1)
    kwargs = dict(initial=initial, grad_output=dy, grad_state=ds,
        normalise=True, chunk_size=2)
    path = args.out / "COMPLETE_SCALAR_GRAPH.jsonl"
    on, receipt = run_trace(path, q, k, v, g, beta, **kwargs)
    off = chunk_vjp(q, k, v, g, beta, **kwargs)
    equality = {"output": on.output.tobytes() == off.output.tobytes(),
        "state": on.state.tobytes() == off.state.tobytes(),
        **{name: on.gradients[name].tobytes() == off.gradients[name].tobytes() for name in NAMES}}
    if not all(equality.values()): raise RuntimeError("OBSERVATION_CHANGED_RESULT")
    reread = validate_trace(path, expected_final_sha256=receipt["final_sha256"])
    result = {"status": "PASS_CPU_COMPLETE_CHUNK_GRAPH_SCOPE", "receipt": receipt,
        "independent_reread": reread, "observer_on_off_bitwise": equality,
        "shape": list(q.shape), "value_dimension": v.shape[-1],
        "trace_bytes": path.stat().st_size, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "synthetic_public_inputs": True, "private_true_data_read": False,
        "full_model_trace": False, "gpu_instruction_trace": False,
        "source_sha256": hashlib.sha256(CHUNK_SOURCE.read_bytes()).hexdigest()}
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
