from pathlib import Path
import copy
import hashlib
import json
import sys

import numpy as np
import pytest

from xiaomai_trace import chunk_trace as trace


def inputs():
    rng = np.random.default_rng(813)
    q, k = [(rng.normal(size=(1, 3, 1, 2)) * .2).astype(np.float32) for _ in range(2)]
    v = (rng.normal(size=(1, 3, 1, 2)) * .3).astype(np.float32)
    g = -rng.uniform(.1, .5, (1, 3, 1)).astype(np.float32)
    beta = rng.uniform(.1, .8, g.shape).astype(np.float32)
    initial = (rng.normal(size=(1, 1, 2, 2)) * .1).astype(np.float32)
    dy = (rng.normal(size=v.shape) * .2).astype(np.float32)
    ds = np.zeros_like(initial)
    return (q, k, v, g, beta), dict(initial=initial, grad_output=dy, grad_state=ds,
        chunk_size=2, normalise=True)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    path = tmp_path_factory.mktemp("full_trace") / "trace.jsonl"
    args, kwargs = inputs()
    result, receipt = trace.run_trace(path, *args, **kwargs)
    return path, result, receipt


def rechain(path, events):
    # All adversarial journals get a valid new hash chain; numeric validation
    # must still reject them, rather than stopping at transport integrity.
    previous = "0" * 64
    with path.open("wb") as stream:
        for seq, event in enumerate(events):
            body = dict(sequence=seq, previous_sha256=previous, event=event)
            previous = hashlib.sha256(trace._canonical(body)).hexdigest()
            stream.write(trace._canonical(dict(**body, sha256=previous)) + b"\n")


def events(path):
    return [json.loads(line)["event"] for line in path.read_text().splitlines()]


def test_independent_all_arithmetic_edges_and_bitwise_observer_off(recorded):
    path, result, receipt = recorded
    args, kwargs = inputs()
    off = trace.chunk_vjp(*args, **kwargs)
    assert result.output.tobytes() == off.output.tobytes()
    assert result.state.tobytes() == off.state.tobytes()
    for name in trace.NAMES:
        assert result.gradients[name].tobytes() == off.gradients[name].tobytes()
    checked = trace.validate_trace(path)
    assert checked["nodes"] == result.receipt["nodes"]
    assert checked["reverse_edges"] == result.receipt["reverse_edges"]
    assert checked["zero_contribution_edges"] > 0
    assert checked["seeds"] == result.output.size + result.state.size
    assert not path.with_suffix(".INCOMPLETE").exists()


@pytest.mark.parametrize("attack", ["delete_node", "duplicate_node", "reorder_node",
    "alter_value", "alter_parent", "alter_derivative", "delete_zero_edge",
    "duplicate_edge", "reorder_edge", "alter_incoming", "alter_before",
    "alter_contribution", "alter_after", "delete_zero_seed", "alter_seed",
    "alter_binding", "alter_gradient", "truncate_footer", "alter_source"])
def test_tamper_rejected_even_with_valid_rebuilt_hash_chain(recorded, tmp_path, attack):
    rows = events(recorded[0])
    forward = [i for i, row in enumerate(rows) if row["phase"] == "forward" and row["parents"]]
    reverse = [i for i, row in enumerate(rows) if row["phase"] == "reverse"]
    zero_edges = [i for i in reverse if float.fromhex(rows[i]["contribution_hex"]) == 0]
    seed = [i for i, row in enumerate(rows) if row["phase"] == "seed"]
    if attack == "delete_node": del rows[forward[0]]
    elif attack == "duplicate_node": rows.insert(forward[0], copy.deepcopy(rows[forward[0]]))
    elif attack == "reorder_node":
        i, j = forward[:2]; rows[i], rows[j] = rows[j], rows[i]
    elif attack == "alter_value": rows[forward[0]]["value_hex"] = (123.0).hex()
    elif attack == "alter_parent": rows[forward[0]]["parents"][0][0] = rows[forward[0]]["node"]
    elif attack == "alter_derivative": rows[forward[0]]["parents"][0][1] = (3.0).hex()
    elif attack == "delete_zero_edge": del rows[zero_edges[0]]
    elif attack == "duplicate_edge": rows.insert(reverse[0], copy.deepcopy(rows[reverse[0]]))
    elif attack == "reorder_edge":
        i, j = reverse[:2]; rows[i], rows[j] = rows[j], rows[i]
    elif attack.startswith("alter_") and attack[6:] in ("incoming", "before", "contribution", "after"):
        rows[reverse[0]][attack[6:] + "_hex"] = (123.0).hex()
    elif attack == "delete_zero_seed": del rows[seed[-1]]
    elif attack == "alter_seed": rows[seed[0]]["value_hex"] = (1.0).hex()
    elif attack == "alter_binding":
        item = next(row for row in rows if row["phase"] == "binding")
        item["state_nodes"][0] = 0
    elif attack == "alter_gradient":
        item = next(row for row in rows if row["phase"] == "returned_gradients")
        item["values"]["q"][0] = (123.0).hex()
    elif attack == "truncate_footer": rows.pop()
    elif attack == "alter_source": rows[0]["source_sha256"] = "0" * 64
    else: raise AssertionError(attack)
    path = tmp_path / (attack + ".jsonl")
    rechain(path, rows)
    with pytest.raises(trace.TraceRejected): trace.validate_trace(path)


def test_observer_failure_preserves_incomplete(tmp_path):
    args, kwargs = inputs()
    path = tmp_path / "observer_failure.jsonl"
    def fail(event):
        if event["phase"] == "forward" and event["node"] == 10:
            raise RuntimeError("OBSERVER_FAILURE")
    with pytest.raises(RuntimeError, match="OBSERVER_FAILURE"):
        trace.run_trace(path, *args, **kwargs, after_event=fail)
    assert path.with_suffix(".INCOMPLETE").exists()
    assert path.with_suffix(".FAILED.json").exists()
    assert not path.with_suffix(".ACCEPTED.json").exists()
    with pytest.raises(trace.TraceRejected): trace.validate_trace(path)


def test_budget_failure_no_acceptance(tmp_path):
    args, kwargs = inputs()
    path = tmp_path / "budget.jsonl"
    with pytest.raises(trace.TraceRejected, match="BYTE_BUDGET"):
        trace.run_trace(path, *args, **kwargs, max_bytes=64)
    assert path.with_suffix(".INCOMPLETE").exists()
    assert not path.with_suffix(".ACCEPTED.json").exists()


def test_source_drift_refused_without_mutating_source(tmp_path, monkeypatch):
    original = trace._sha_file
    calls = []
    def drift(path):
        calls.append(path)
        return original(path) if len(calls) == 1 else "0" * 64
    monkeypatch.setattr(trace, "_sha_file", drift)
    args, kwargs = inputs()
    path = tmp_path / "drift.jsonl"
    with pytest.raises(trace.TraceRejected, match="SOURCE_CHANGED"):
        trace.run_trace(path, *args, **kwargs)
    assert path.with_suffix(".INCOMPLETE").exists()
    assert not path.with_suffix(".ACCEPTED.json").exists()


def test_transport_reorder_or_deleted_rows_rejected(recorded, tmp_path):
    rows = recorded[0].read_bytes().splitlines(keepends=True)
    rows[4], rows[5] = rows[5], rows[4]
    path = tmp_path / "raw_reorder.jsonl"
    path.write_bytes(b"".join(rows))
    with pytest.raises(trace.TraceRejected): trace.validate_trace(path)


def test_refuse_existing_trace_overwrite(recorded):
    args, kwargs = inputs()
    with pytest.raises(trace.TraceRejected, match="OVERWRITE"):
        trace.run_trace(recorded[0], *args, **kwargs)


def test_trusted_external_receipt_identifies_original_even_for_self_consistent_rechain(recorded, tmp_path):
    rows = events(recorded[0])
    rows[0]["experiment_label"] = "different_identifiable_experiment"
    path = tmp_path / "other_identity.jsonl"
    rechain(path, rows)
    # Adding nonnumeric metadata does not invalidate the graph mathematics.
    assert trace.validate_trace(path)["arithmetic_recomputed"]
    with pytest.raises(trace.TraceRejected, match="TRUSTED_RECEIPT_ROOT"):
        trace.validate_trace(path, expected_final_sha256=recorded[2]["final_sha256"])
