"""Readable CPU VJP of the *chunk* GDN triangular-solve algorithm.

This replays the published v094 chunk finite-precision forward, never substitutes
the recurrent algorithm and never calls native backward/matmul. Derivatives use
the conventional straight-through derivative for FP32 rounding, as torch does.
Local derivatives use the actual rounded forward operands; reverse accumulation
is binary64, followed by an explicit FP32 cast at each returned input gradient.
This deliberately small, budgeted scalar graph is a correctness/trace path, not
a claim of full-size throughput, BF16, CUDA, or complete model training support.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable

import numpy as np

NAMES = ("q", "k", "v", "g", "beta", "initial")


class ChunkDomainError(ValueError):
    pass


def _sha(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


@dataclass
class ChunkResult:
    output: np.ndarray
    state: np.ndarray
    gradients: dict[str, np.ndarray]
    receipt: dict


class _Graph:
    """Every operation stores its actual forward value and local derivatives."""
    def __init__(self, max_nodes, observer=None, max_edges=None, deadline=None):
        self.values = []
        self.parents = []
        self.ops = []
        self.max_nodes = max_nodes
        self.observer = observer
        self.coordinate = None
        self.constants = {}
        self.max_edges, self.edges, self.deadline = max_edges, 0, deadline

    def node(self, value, parents=(), op="leaf"):
        if len(self.values) >= self.max_nodes:
            raise ChunkDomainError("SCALAR_GRAPH_NODE_BUDGET_EXCEEDED")
        self.edges += len(parents)
        if self.max_edges is not None and self.edges > self.max_edges:
            raise ChunkDomainError("GRAPH_EDGE_BUDGET_EXCEEDED")
        if self.deadline is not None and len(self.values) % 1024 == 0 and time.perf_counter() > self.deadline:
            raise ChunkDomainError("CHUNK_WALL_TIME_BUDGET_EXCEEDED")
        value = float(value)
        if not math.isfinite(value):
            raise ChunkDomainError(f"FIRST_NONFINITE_FORWARD:{op}:{self.coordinate}")
        n = len(self.values)
        if self.observer is not None:
            self.observer({"phase": "forward", "node": n, "op": op,
                "coordinate": self.coordinate, "value_hex": value.hex(),
                "parents": tuple((p, d.hex()) for p, d in parents)})
        self.values.append(value)
        self.parents.append(parents)
        self.ops.append(op)
        return n

    def const(self, value):
        value = float(value)
        # Preserve +0 / -0 constants rather than collapsing equal values.
        key = value.hex()
        if key not in self.constants:
            self.constants[key] = self.node(value, op="constant")
        return self.constants[key]

    def add(self, a, b, fp32=True):
        value = self.values[a] + self.values[b]
        return self.node(np.float32(value) if fp32 else value,
            ((a, 1.0), (b, 1.0)), "add32" if fp32 else "add64")

    def sub(self, a, b):
        return self.node(np.float32(self.values[a] - self.values[b]),
            ((a, 1.0), (b, -1.0)), "sub32")

    def neg(self, a):
        return self.node(np.float32(-self.values[a]), ((a, -1.0),), "neg32")

    def mul(self, a, b, fp32=True):
        x, y = self.values[a], self.values[b]
        return self.node(np.float32(x * y) if fp32 else x * y,
            ((a, y), (b, x)), "mul32" if fp32 else "mul64")

    def cast(self, a):
        return self.node(np.float32(self.values[a]), ((a, 1.0),), "round32")

    def exp(self, a):
        x = self.values[a]
        value = np.float32(math.exp(x)) if x <= 709.0 else np.float32(np.inf)
        return self.node(value, ((a, float(value)),), "exp32")

    def sqrt(self, a):
        value = np.float32(math.sqrt(self.values[a]))
        return self.node(value, ((a, 0.5 / float(value)),), "sqrt32")

    def div(self, a, b):
        x, y = self.values[a], self.values[b]
        if y == 0.0:
            raise ChunkDomainError(f"ZERO_DIVISOR:{self.coordinate}")
        return self.node(np.float32(x / y), ((a, 1.0 / y), (b, -x / (y * y))), "div32")

    def reverse(self, seeds):
        cotangents = [0.0] * len(self.values)
        for n, value in seeds:
            cotangents[n] += float(value)
        for n in range(len(self.values) - 1, -1, -1):
            if self.deadline is not None and n % 1024 == 0 and time.perf_counter() > self.deadline:
                raise ChunkDomainError("CHUNK_WALL_TIME_BUDGET_EXCEEDED")
            incoming = cotangents[n]
            for parent, derivative in self.parents[n]:
                before = cotangents[parent]
                contribution = incoming * derivative
                after = before + contribution
                if not math.isfinite(after):
                    raise ChunkDomainError(f"FIRST_NONFINITE_REVERSE:{n}:{parent}")
                if self.observer is not None:
                    self.observer({"phase": "reverse", "node": n, "parent": parent,
                        "incoming_hex": incoming.hex(), "derivative_hex": derivative.hex(),
                        "contribution_hex": contribution.hex(), "before_hex": before.hex(),
                        "after_hex": after.hex()})
                cotangents[parent] = after
        return cotangents


def _mm(graph, a, b):
    # Original chunk.py computes each dot in binary64 then rounds once to FP32.
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    output = []
    for i in range(rows):
        row = []
        for j in range(cols):
            total = graph.const(0.0)
            for p in range(inner):
                total = graph.add(total, graph.mul(a[i][p], b[p][j], False), False)
            row.append(graph.cast(total))
        output.append(row)
    return output


def _transpose(a):
    return [list(row) for row in zip(*a)]


def _mm_compact(graph, a, b):
    """Contract binary64 dot edges; retain exact forward order and operands.

    No BLAS/autograd. Dot-internal add/multiply nodes are omitted, so this is
    explicitly NOT a complete scalar trace. Straight-through rounding gives
    the same real derivative; reverse accumulation order may differ in FP64.
    """
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    output = []
    for i in range(rows):
        row = []
        for j in range(cols):
            total, parents = 0.0, []
            for p in range(inner):
                ai, bj = a[i][p], b[p][j]
                x, y = graph.values[ai], graph.values[bj]
                total = total + x * y
                parents.extend(((ai, y), (bj, x)))
            row.append(graph.node(np.float32(total), tuple(parents), "contracted_dot64_round32"))
        output.append(row)
    return output


def _validate(q, k, v, g, beta, initial, grad_output, grad_state, C, max_nodes):
    arrays = (q, k, v, g, beta)
    if any(type(a) is not np.ndarray or a.dtype != np.float32 for a in arrays):
        raise ChunkDomainError("EXACT_CPU_NUMPY_FP32_REQUIRED")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ChunkDomainError("QKV_SHAPE")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if min(B, T, H, K, V) <= 0 or g.shape != (B, T, H) or beta.shape != g.shape:
        raise ChunkDomainError("EMPTY_OR_GATE_SHAPE")
    if type(C) is not int or not 1 <= C <= 128:
        raise ChunkDomainError("CHUNK_SIZE_RANGE")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 1_000_000:
        raise ChunkDomainError("GRAPH_NODE_BUDGET_RANGE_1_TO_1000000")
    if initial is None:
        initial = np.zeros((B, H, K, V), np.float32)
    if type(initial) is not np.ndarray or initial.dtype != np.float32 or initial.shape != (B, H, K, V):
        raise ChunkDomainError("INITIAL_SHAPE_OR_DTYPE")
    if grad_output is None:
        grad_output = np.zeros(v.shape, np.float32)
    if grad_state is None:
        grad_state = np.zeros(initial.shape, np.float32)
    if any(type(a) is not np.ndarray or a.dtype != np.float32 for a in (grad_output, grad_state)):
        raise ChunkDomainError("COTANGENT_FP32_REQUIRED")
    if grad_output.shape != v.shape or grad_state.shape != initial.shape:
        raise ChunkDomainError("COTANGENT_SHAPE")
    arrays += (initial, grad_output, grad_state)
    if sum(a.nbytes for a in arrays) > 16 * 1024**2:
        raise ChunkDomainError("DIAGNOSTIC_INPUT_BUDGET_16_MIB")
    if any(any(s < 0 for s in a.strides) for a in arrays):
        raise ChunkDomainError("NEGATIVE_STRIDES_UNSUPPORTED")
    for name, a in zip(NAMES + ("grad_output", "grad_state"), arrays):
        if not np.isfinite(a).all():
            where = tuple(int(i) for i in np.argwhere(~np.isfinite(a))[0])
            raise ChunkDomainError(f"NONFINITE_INPUT:{name}:{where}")
    return initial, grad_output, grad_state


def chunk_vjp(q, k, v, g, beta, *, initial=None, grad_output=None, grad_state=None,
              normalise=False, chunk_size=4, max_nodes=400_000,
              observer: Callable[[dict], None] | None = None,
              compact_dots=False, max_edges=None, max_seconds=None):
    """Return original chunk forward and all six input VJPs.

    Both output and final-state cotangents are seeded on the *same* graph. Zero
    cotangents and contributions are retained by the optional observer. Raising
    from the observer aborts the result; missing log events are never ignored.
    ``None`` initial state means a zero initial state; its diagnostic derivative
    is still returned under ``initial``. This is a numeric interface; a caller
    must explicitly register it on its Tape before claiming model integration.
    """
    if type(normalise) is not bool:
        raise ChunkDomainError("EXACT_BOOLEAN_NORMALISE_REQUIRED")
    if type(compact_dots) is not bool:
        raise ChunkDomainError("EXACT_BOOLEAN_COMPACT_DOTS_REQUIRED")
    if max_edges is not None and (type(max_edges) is not int or not 1 <= max_edges <= 8_000_000):
        raise ChunkDomainError("EDGE_BUDGET_RANGE_1_TO_8000000")
    if max_seconds is not None and (type(max_seconds) not in (int, float) or not 0 < max_seconds <= 120):
        raise ChunkDomainError("WALL_TIME_BUDGET_RANGE_0_TO_120")
    started = time.perf_counter()
    initial, dy, ds = _validate(q, k, v, g, beta, initial, grad_output, grad_state,
        chunk_size, max_nodes)
    arrays = (q, k, v, g, beta, initial)
    before = tuple(_sha(a) for a in arrays)
    graph = _Graph(max_nodes, observer, max_edges, None if max_seconds is None else started + max_seconds)
    mm = _mm_compact if compact_dots else _mm
    zero, one = graph.const(0.0), graph.const(1.0)
    ids = []
    for a in arrays:
        entry = np.empty(a.shape, np.int64)
        for position in np.ndindex(a.shape):
            graph.coordinate = ("input", len(ids), position)
            entry[position] = graph.node(a[position])
        ids.append(entry)
    qi, ki, vi, gi, bi, si = ids
    B, T, H, K = q.shape
    V, C = v.shape[-1], chunk_size
    scale = graph.const(np.float32(1.0 / math.sqrt(K)))
    if normalise:
        eps = graph.const(np.float32(1e-6))
        for b in range(B):
            for t in range(T):
                for h in range(H):
                    for which, target in enumerate((qi, ki)):
                        graph.coordinate = ("normalise", b, t, h, which)
                        total = zero
                        for j in range(K):
                            n = int(target[b, t, h, j])
                            total = graph.add(total, graph.mul(n, n))
                        inverse = graph.div(one, graph.sqrt(graph.add(total, eps)))
                        for j in range(K):
                            target[b, t, h, j] = graph.mul(int(target[b, t, h, j]), inverse)
    output_ids = np.empty(v.shape, np.int64)
    state_ids = si.copy()
    for b in range(B):
        for h in range(H):
            for start in range(0, T, C):
                graph.coordinate = ("chunk", b, h, start)
                qc, kc, kb, vb, gc = [], [], [], [], []
                acc = zero
                for i in range(C):
                    live, t = start + i < T, start + i
                    gate, bet = (int(gi[b, t, h]), int(bi[b, t, h])) if live else (zero, zero)
                    acc = graph.add(acc, gate, False)
                    gc.append(graph.cast(acc))
                    qr, kr, kbr, vr = [], [], [], []
                    for j in range(K):
                        qn, kn = (int(qi[b, t, h, j]), int(ki[b, t, h, j])) if live else (zero, zero)
                        qr.append(graph.mul(qn, scale))
                        kr.append(kn)
                        kbr.append(graph.mul(kn, bet))
                    for j in range(V):
                        vn = int(vi[b, t, h, j]) if live else zero
                        vr.append(graph.mul(vn, bet))
                    qc.append(qr); kc.append(kr); kb.append(kbr); vb.append(vr)
                prod = mm(graph, kb, _transpose(kc))
                decay, a = [], []
                for i in range(C):
                    dr, ar = [], []
                    for j in range(C):
                        d = graph.exp(graph.sub(gc[i], gc[j])) if i >= j else zero
                        dr.append(d)
                        ar.append(graph.neg(graph.mul(prod[i][j], d)) if i > j else zero)
                    decay.append(dr); a.append(ar)
                for i in range(1, C):
                    row = a[i][:i].copy()
                    for j in range(i):
                        total = zero
                        for p in range(i):
                            total = graph.add(total, graph.mul(row[p], a[p][j]), False)
                        a[i][j] = graph.add(row[j], graph.cast(total))
                for i in range(C):
                    a[i][i] = graph.add(a[i][i], one)
                vc = mm(graph, a, vb)
                scaled_k, scaled_q = [], []
                for i in range(C):
                    e = graph.exp(gc[i])
                    scaled_k.append([graph.mul(n, e) for n in kb[i]])
                    scaled_q.append([graph.mul(n, e) for n in qc[i]])
                cumulative = mm(graph, a, scaled_k)
                previous = [[int(state_ids[b, h, i, j]) for j in range(V)] for i in range(K)]
                memory = mm(graph, cumulative, previous)
                new_value = [[graph.sub(vc[i][j], memory[i][j]) for j in range(V)] for i in range(C)]
                attn = mm(graph, qc, _transpose(kc))
                for i in range(C):
                    for j in range(C):
                        attn[i][j] = graph.mul(attn[i][j], decay[i][j])
                from_past, from_chunk = mm(graph, scaled_q, previous), mm(graph, attn, new_value)
                for i in range(min(C, T - start)):
                    for j in range(V):
                        output_ids[b, start + i, h, j] = graph.add(from_past[i][j], from_chunk[i][j])
                weighted_k = []
                for i in range(C):
                    e = graph.exp(graph.sub(gc[-1], gc[i]))
                    weighted_k.append([graph.mul(n, e) for n in kc[i]])
                increment, e = mm(graph, _transpose(weighted_k), new_value), graph.exp(gc[-1])
                for i in range(K):
                    for j in range(V):
                        state_ids[b, h, i, j] = graph.add(graph.mul(previous[i][j], e), increment[i][j])
    output = np.asarray([graph.values[int(n)] for n in output_ids.flat], np.float32).reshape(v.shape)
    state = np.asarray([graph.values[int(n)] for n in state_ids.flat], np.float32).reshape(initial.shape)
    seeds = [(int(n), float(c)) for n, c in zip(output_ids.flat, dy.flat)]
    seeds += [(int(n), float(c)) for n, c in zip(state_ids.flat, ds.flat)]
    if observer is not None:
        observer({"phase": "binding", "output_nodes": [int(n) for n in output_ids.flat],
            "state_nodes": [int(n) for n in state_ids.flat],
            "output_shape": output.shape, "state_shape": state.shape})
        for ordinal, (n, value) in enumerate(seeds):
            observer({"phase": "seed", "ordinal": ordinal, "node": n,
                "value_hex": value.hex()})
    cotangents = graph.reverse(seeds)
    # q/k node arrays were rebound during normalization; leaf IDs remain a
    # contiguous prefix in creation order. Return gradients of original leaves.
    gradients, cursor = {}, 2
    for name, original in zip(NAMES, arrays):
        count = original.size
        values = np.asarray(cotangents[cursor:cursor + count], np.float32).reshape(original.shape)
        if not np.isfinite(values).all():
            raise ChunkDomainError("NONFINITE_RETURNED_GRADIENT:" + name)
        gradients[name] = values
        cursor += count
    if observer is not None:
        observer({"phase": "returned_gradients", "values": {
            name: [float(v).hex() for v in gradients[name].flat] for name in NAMES}})
    if before != tuple(_sha(a) for a in arrays):
        raise ChunkDomainError("INPUT_CHANGED_DURING_CALL")
    return ChunkResult(output, state, gradients, {
        "algorithm": "PYTHON_CHUNK_TRIANGULAR_SOLVE_SCALAR_VJP",
        "native_backward_used": False, "native_blas_used": False,
        "actual_gpu": False, "dtype": "CPU_FP32", "chunk_size": C,
        "normalise": normalise, "nodes": len(graph.values),
        "reverse_edges": sum(len(p) for p in graph.parents),
        "input_sha256": dict(zip(NAMES, before)), "seconds": time.perf_counter() - started,
        "forward_precision": "FP32_ELEMENTWISE_BINARY64_DOT_THEN_FP32",
        "reverse_precision": "BINARY64_ACCUMULATION_THEN_FP32_RETURN",
        "rounding_derivative": "STRAIGHT_THROUGH",
        "output_and_final_state_same_graph": True,
        "observer_enabled": observer is not None,
        "compact_dots": compact_dots, "dot_internal_scalar_trace_complete": not compact_dots,
        "full_model_integrated": False, "native_boundary": ["NumPy allocation and FP32 casts", "libm exp/sqrt"],
    })
