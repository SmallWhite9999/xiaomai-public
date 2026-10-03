"""Budgeted, complete scalar chunk graph journal and independent numeric reader.

The reader recomputes each arithmetic result, local derivative and every reverse
edge (including zero contributions). Hashes bind sequence/custody; they are not
the numerical validation. Scope: the CPU chunk scalar graph, not full True or
GPU hardware instructions. Input values in examples must be synthetic/public.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SOURCE = ROOT / "chunk_training/chunk_vjp.py"
from chunk_training import chunk_vjp

NAMES = ("q", "k", "v", "g", "beta", "initial")


class TraceRejected(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _f32(value):
    # Independent of the producer's NumPy cast.
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _hex(value):
    return float(value).hex()


def _float(word):
    value = float.fromhex(word)
    if not math.isfinite(value) or value.hex() != word:
        raise TraceRejected("NONFINITE_OR_NONCANONICAL_HEX")
    return value


def _array(a):
    return {"shape": list(a.shape), "values": [_hex(v) for v in a.flat],
        "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def _count(shape):
    result = 1
    for size in shape:
        if type(size) is not int or size <= 0:
            raise TraceRejected("INVALID_SHAPE")
        result *= size
    return result


class Journal:
    def __init__(self, path, max_bytes):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise TraceRejected("REFUSE_OVERWRITE_EXISTING_JOURNAL")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 512 * 1024**2:
            raise TraceRejected("JOURNAL_BYTE_BUDGET_RANGE")
        self.marker = self.path.with_suffix(".INCOMPLETE")
        self.marker.write_text("INCOMPLETE\n", encoding="utf-8")
        self.stream = self.path.open("xb")
        self.max_bytes, self.bytes, self.sequence = max_bytes, 0, 0
        self.previous = "0" * 64

    def append(self, event):
        body = {"sequence": self.sequence, "previous_sha256": self.previous,
            "event": event}
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        encoded = _canonical({**body, "sha256": digest}) + b"\n"
        if self.bytes + len(encoded) > self.max_bytes:
            raise TraceRejected("JOURNAL_BYTE_BUDGET_EXCEEDED")
        self.stream.write(encoded)
        self.stream.flush()
        self.bytes += len(encoded)
        self.sequence += 1
        self.previous = digest

    def close(self):
        self.stream.close()


def run_trace(path, q, k, v, g, beta, *, initial, grad_output, grad_state,
              chunk_size=2, normalise=False, max_bytes=64 * 1024**2,
              after_event=None):
    """A failure leaves INCOMPLETE and a FAILED receipt; never an acceptance."""
    path = Path(path)
    journal = Journal(path, max_bytes)
    source_sha = _sha_file(CHUNK_SOURCE)
    inputs = (q, k, v, g, beta, initial)
    contract = {"phase": "header", "schema": "CHUNK_SCALAR_TRACE_V1",
        "source_sha256": source_sha, "scope": "CPU_CHUNK_SCALAR_GRAPH",
        "inputs": {name: _array(a) for name, a in zip(NAMES, inputs)},
        "grad_output": _array(grad_output), "grad_state": _array(grad_state),
        "chunk_size": chunk_size, "normalise": normalise}
    try:
        journal.append(contract)
        def observer(event):
            journal.append(event)
            if after_event is not None:
                after_event(event)
        result = chunk_vjp(q, k, v, g, beta, initial=initial,
            grad_output=grad_output, grad_state=grad_state,
            chunk_size=chunk_size, normalise=normalise, observer=observer)
        if _sha_file(CHUNK_SOURCE) != source_sha:
            raise TraceRejected("SOURCE_CHANGED_DURING_TRACE")
        journal.append({"phase": "footer", "output": _array(result.output),
            "state": _array(result.state), "nodes": result.receipt["nodes"],
            "reverse_edges": result.receipt["reverse_edges"]})
        journal.close()
        receipt = validate_trace(path, expected_source_sha256=source_sha)
        receipt.update({"status": "ACCEPTED_COMPLETE_NUMERIC_SCOPE",
            "actual_gpu": False, "current_true_restored": False,
            "full_model_trace": False})
        path.with_suffix(".ACCEPTED.json").write_text(json.dumps(receipt,
            ensure_ascii=False, indent=2), encoding="utf-8")
        journal.marker.unlink()
        return result, receipt
    except BaseException as exc:
        journal.close()
        path.with_suffix(".FAILED.json").write_text(json.dumps({"status": "FAILED",
            "error": str(exc), "events_written": journal.sequence,
            "incomplete": True}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _arithmetic(event, values):
    op = event["op"]
    parents = [(int(n), _float(d)) for n, d in event["parents"]]
    if any(n < 0 or n >= len(values) for n, _ in parents):
        raise TraceRejected("PARENT_NOT_EARLIER_NODE")
    x = [values[n] for n, _ in parents]
    if op in ("leaf", "constant"):
        if parents:
            raise TraceRejected("LEAF_HAS_PARENTS")
        return _float(event["value_hex"]), []
    if op in ("add32", "add64", "sub32", "mul32", "mul64", "div32"):
        if len(x) != 2:
            raise TraceRejected("BINARY_ARITY")
        a, b = x
        if op.startswith("add"):
            result, ds = a + b, [1.0, 1.0]
        elif op == "sub32":
            result, ds = a - b, [1.0, -1.0]
        elif op.startswith("mul"):
            result, ds = a * b, [b, a]
        else:
            result, ds = a / b, [1.0 / b, -a / (b * b)]
    else:
        if len(x) != 1:
            raise TraceRejected("UNARY_ARITY")
        a = x[0]
        if op == "neg32": result, ds = -a, [-1.0]
        elif op == "round32": result, ds = a, [1.0]
        elif op == "exp32": result, ds = math.exp(a), None
        elif op == "sqrt32": result, ds = math.sqrt(a), None
        else: raise TraceRejected("UNSUPPORTED_OPERATION:" + op)
    if op.endswith("32"):
        result = _f32(result)
    if op == "exp32": ds = [result]
    if op == "sqrt32": ds = [0.5 / result]
    if [_hex(d) for _, d in parents] != [_hex(d) for d in ds]:
        raise TraceRejected("LOCAL_DERIVATIVE_MISMATCH")
    return result, parents


def validate_trace(path, *, expected_source_sha256=None, expected_final_sha256=None,
                   max_bytes=512 * 1024**2):
    """Read the journal independently and recompute numbers and full edge coverage.

    A caller retaining a trusted acceptance receipt should also check its final
    root. This validator establishes this recorded graph's arithmetic; the
    pinned algorithm's correctness is separately tested against torch/old chunk.
    """
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise TraceRejected("READER_BYTE_BUDGET_EXCEEDED")
    source_sha = expected_source_sha256 or _sha_file(CHUNK_SOURCE)
    values, parent_edges, operations = [], [], []
    binding, contract, footer = None, None, None
    cotangents = None
    stage, previous, sequence, seeds_seen, edges_seen = "header", "0" * 64, 0, 0, 0
    expected_edges = None
    zero_edges, seen_return = 0, False
    with path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            digest = row.pop("sha256")
            if row.get("sequence") != sequence or row.get("previous_sha256") != previous:
                raise TraceRejected("SEQUENCE_OR_CHAIN_MISMATCH")
            if hashlib.sha256(_canonical(row)).hexdigest() != digest:
                raise TraceRejected("EVENT_HASH_MISMATCH")
            previous, sequence = digest, sequence + 1
            e, phase = row["event"], row["event"]["phase"]
            if phase == "header":
                if stage != "header" or contract is not None or sequence != 1:
                    raise TraceRejected("HEADER_ORDER")
                contract = e
                if e.get("schema") != "CHUNK_SCALAR_TRACE_V1" or e["source_sha256"] != source_sha:
                    raise TraceRejected("SOURCE_OR_SCHEMA_MISMATCH")
                if set(e["inputs"]) != set(NAMES):
                    raise TraceRejected("INPUT_NAMES")
                for item in list(e["inputs"].values()) + [e["grad_output"], e["grad_state"]]:
                    if len(item["values"]) != _count(item["shape"]):
                        raise TraceRejected("ARRAY_COUNT_MISMATCH")
                    raw = b"".join(struct.pack("<f", _float(w)) for w in item["values"])
                    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                        raise TraceRejected("ARRAY_CONTENT_MISMATCH")
                stage = "forward"
            elif phase == "forward":
                if stage != "forward" or e["node"] != len(values):
                    raise TraceRejected("FORWARD_NODE_ORDER")
                value, parents = _arithmetic(e, values)
                if _hex(value) != e["value_hex"]:
                    raise TraceRejected("FORWARD_ARITHMETIC_MISMATCH")
                if e["op"] == "leaf":
                    coord = e["coordinate"]
                    if coord[0] != "input" or not 0 <= coord[1] < 6:
                        raise TraceRejected("INPUT_COORDINATE")
                    item = contract["inputs"][NAMES[coord[1]]]
                    flat = int(np.ravel_multi_index(tuple(coord[2]), tuple(item["shape"])))
                    if e["value_hex"] != item["values"][flat]:
                        raise TraceRejected("INPUT_VALUE_MISMATCH")
                    expected_node = 2 + sum(_count(contract["inputs"][name]["shape"])
                        for name in NAMES[:coord[1]]) + flat
                    if e["node"] != expected_node:
                        raise TraceRejected("INPUT_COORDINATE_ORDER_OR_DUPLICATE")
                values.append(value); parent_edges.append(parents); operations.append(e["op"])
            elif phase == "binding":
                if stage != "forward" or binding is not None:
                    raise TraceRejected("BINDING_ORDER")
                binding = e
                if operations[:2] != ["constant", "constant"] or [_hex(v) for v in values[:2]] != [_hex(0), _hex(1)]:
                    raise TraceRejected("INITIAL_CONSTANT_BINDING")
                for prefix, key in (("output", "grad_output"), ("state", "grad_state")):
                    if e[prefix + "_shape"] != contract[key]["shape"]:
                        raise TraceRejected("BINDING_SHAPE")
                    if len(e[prefix + "_nodes"]) != _count(e[prefix + "_shape"]):
                        raise TraceRejected("BINDING_NODE_COUNT")
                    if any(type(n) is not int or not 0 <= n < len(values) for n in e[prefix + "_nodes"]):
                        raise TraceRejected("BINDING_NODE_RANGE")
                cotangents = [0.0] * len(values)
                stage = "seed"
            elif phase == "seed":
                if stage != "seed" or e["ordinal"] != seeds_seen:
                    raise TraceRejected("SEED_ORDER")
                nodes = binding["output_nodes"] + binding["state_nodes"]
                words = contract["grad_output"]["values"] + contract["grad_state"]["values"]
                if seeds_seen >= len(nodes) or e["node"] != nodes[seeds_seen] or e["value_hex"] != words[seeds_seen]:
                    raise TraceRejected("SEED_CONTENT_MISMATCH")
                cotangents[e["node"]] += _float(e["value_hex"])
                seeds_seen += 1
            elif phase == "reverse":
                if stage == "seed":
                    if seeds_seen != len(binding["output_nodes"]) + len(binding["state_nodes"]):
                        raise TraceRejected("MISSING_SEED")
                    expected_edges = [(n, p, d) for n in range(len(values) - 1, -1, -1) for p, d in parent_edges[n]]
                    stage = "reverse"
                if stage != "reverse" or edges_seen >= len(expected_edges):
                    raise TraceRejected("REVERSE_ORDER_OR_EXTRA_EDGE")
                n, p, d = expected_edges[edges_seen]
                incoming, before = cotangents[n], cotangents[p]
                contribution, after = incoming * d, before + incoming * d
                expected = {"node": n, "parent": p, "incoming_hex": _hex(incoming),
                    "derivative_hex": _hex(d), "contribution_hex": _hex(contribution),
                    "before_hex": _hex(before), "after_hex": _hex(after)}
                if any(e.get(key) != value for key, value in expected.items()):
                    raise TraceRejected("REVERSE_NUMERICAL_OR_EDGE_MISMATCH")
                cotangents[p] = after
                zero_edges += int(contribution == 0.0)
                edges_seen += 1
            elif phase == "returned_gradients":
                if stage != "reverse" or edges_seen != len(expected_edges) or seen_return:
                    raise TraceRejected("MISSING_REVERSE_EDGE_OR_RETURN_ORDER")
                cursor = 2
                if set(e["values"]) != set(NAMES):
                    raise TraceRejected("GRADIENT_NAMES")
                for name in NAMES:
                    count = _count(contract["inputs"][name]["shape"])
                    if operations[cursor:cursor + count] != ["leaf"] * count:
                        raise TraceRejected("LEAF_BINDING_MISMATCH")
                    expected = [_hex(_f32(v)) for v in cotangents[cursor:cursor + count]]
                    if e["values"][name] != expected:
                        raise TraceRejected("RETURNED_GRADIENT_MISMATCH")
                    cursor += count
                seen_return, stage = True, "footer"
            elif phase == "footer":
                if stage != "footer" or footer is not None:
                    raise TraceRejected("FOOTER_ORDER")
                if e["nodes"] != len(values) or e["reverse_edges"] != edges_seen:
                    raise TraceRejected("COUNT_MISMATCH")
                for prefix in ("output", "state"):
                    item = e[prefix]
                    if item["shape"] != binding[prefix + "_shape"] or item["values"] != [_hex(_f32(values[n])) for n in binding[prefix + "_nodes"]]:
                        raise TraceRejected("OUTPUT_STATE_BINDING_MISMATCH")
                    raw = b"".join(struct.pack("<f", _float(v)) for v in item["values"])
                    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                        raise TraceRejected("OUTPUT_STATE_BYTES_MISMATCH")
                footer, stage = e, "done"
            else:
                raise TraceRejected("UNKNOWN_PHASE")
    if stage != "done" or footer is None:
        raise TraceRejected("INCOMPLETE_TRACE")
    if expected_final_sha256 is not None and previous != expected_final_sha256:
        raise TraceRejected("TRUSTED_RECEIPT_ROOT_MISMATCH")
    return {"status": "NUMERIC_GRAPH_VALIDATED", "events": sequence,
        "nodes": len(values), "reverse_edges": edges_seen,
        "zero_contribution_edges": zero_edges, "seeds": seeds_seen,
        "source_sha256": source_sha, "final_sha256": previous,
        "arithmetic_recomputed": True, "all_reverse_edges_recomputed": True,
        "output_and_final_state_seeded_same_graph": True}
