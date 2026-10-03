"""Reference components for a proposed research core; NOT an Ornith integration.

The author supplies the operators, finite models, features and test fixtures.
This module does not discover scientific questions, prove empirical truth, or
train a 9B model. It makes selected mathematical contracts executable.
Only built-in expression operators are interpreted; arbitrary code is not run.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def information_gain(prior: Tensor, likelihood: Tensor) -> Tensor:
    """Return I(H;Y | a,D), in nats, for each action.

    prior: [H]; likelihood: [A,H,O], conditional on the supplied model class.
    No guarantee is made that the class contains reality or that probabilities
    are calibrated. Identical noisy predictions have zero information gain.
    """
    if prior.ndim != 1 or likelihood.ndim != 3:
        raise ValueError('expected [H] and [A,H,O]')
    if min(likelihood.shape) < 1 or prior.numel() != likelihood.shape[1]:
        raise ValueError('empty/mismatched model class')
    if prior.device != likelihood.device:
        raise ValueError('prior and likelihood must share a device')
    p, q = prior.to(torch.float64), likelihood.to(torch.float64)
    for name, value in [('prior', p), ('likelihood', q)]:
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(name + ' must be finite and nonnegative')
    if not torch.allclose(p.sum(), p.new_tensor(1.), atol=1e-8, rtol=1e-7):
        raise ValueError('prior is not normalized')
    if not torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-8, rtol=1e-7):
        raise ValueError('likelihood rows are not normalized')
    predictive = torch.einsum('h,aho->ao', p, q)
    marginal_entropy = -torch.xlogy(predictive, predictive).sum(-1)
    within_entropy = -torch.xlogy(q, q).sum(-1)
    return marginal_entropy - (within_entropy * p[None]).sum(-1)


# Exponents of mass, length, time. This example intentionally does NOT impose
# a physics prior on nonphysical quantities or equate nats with joules.
Unit = tuple[int, int, int]
SCALAR: Unit = (0, 0, 0)
LENGTH: Unit = (0, 1, 0)
TIME: Unit = (0, 0, 1)


@dataclass(frozen=True)
class Expr:
    """A bounded, typed expression tree; a definition, not an empirical law."""
    op: str
    args: tuple['Expr', ...] = ()
    name: str | None = None
    unit: Unit = SCALAR

    def __post_init__(self):
        if len(self.unit) != 3 or any(type(v) is not int for v in self.unit):
            raise ValueError('unit must contain three integer exponents')
        if self.op == 'var':
            if self.args or not isinstance(self.name, str) or not self.name:
                raise ValueError('invalid variable')
            return
        if self.op not in ('add', 'sub', 'mul', 'div') or len(self.args) != 2 or self.name is not None:
            raise ValueError('invalid operator')
        a, b = self.args
        if self.op in ('add', 'sub'):
            if a.unit != b.unit or self.unit != a.unit:
                raise ValueError('addition/subtraction requires equal units')
        else:
            sign = 1 if self.op == 'mul' else -1
            expected = tuple(x + sign*y for x, y in zip(a.unit, b.unit))
            if self.unit != expected:
                raise ValueError('invalid product/quotient unit')

    @staticmethod
    def variable(name: str, unit: Unit = SCALAR) -> 'Expr':
        return Expr('var', name=name, unit=unit)

    def combine(self, op: str, other: 'Expr') -> 'Expr':
        if op in ('add', 'sub'):
            unit = self.unit
        elif op in ('mul', 'div'):
            sign = 1 if op == 'mul' else -1
            unit = tuple(x + sign*y for x, y in zip(self.unit, other.unit))
        else:
            raise ValueError('unknown operator')
        return Expr(op, (self, other), unit=unit)

    def evaluate(self, values: Mapping[str, float]) -> float:
        if self.op == 'var':
            result = float(values[self.name])
        else:
            a, b = (part.evaluate(values) for part in self.args)
            if self.op == 'add': result = a + b
            elif self.op == 'sub': result = a - b
            elif self.op == 'mul': result = a * b
            else:
                if b == 0: raise ZeroDivisionError('definition outside domain')
                result = a / b
        if not math.isfinite(result):
            raise ValueError('nonfinite expression result')
        return result

    def expanded_node_count(self) -> int:
        """A demonstrative structural cost, NOT an optimal MDL code length."""
        return 1 + sum(a.expanded_node_count() for a in self.args)


@dataclass(frozen=True)
class Concept:
    name: str
    expression: Expr
    applicability: str
    version: int = 1

    @property
    def structural_cost(self) -> int:
        # Renaming a large expression cannot hide its implementation cost.
        return self.expression.expanded_node_count()


class ResearchStateCore(nn.Module):
    """Event-level latent recurrence with variable-length candidate scoring.

    Inputs are ALREADY ENCODED evidence and candidate features. Correct semantic
    encoding and candidate-program generation are NOT implemented here. State
    is explicit; it is not a KV cache. No state is implicitly carried between
    unrelated runs. A caller must version and checkpoint it explicitly.
    """
    def __init__(self, width: int, slots: int, outcome_classes: int):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (width, slots, outcome_classes)):
            raise ValueError('positive integer dimensions required')
        self.width, self.slots = width, slots
        self.initial = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(2*width, width)
        self.proposal = nn.Sequential(nn.Linear(2*width, 2*width), nn.SiLU(), nn.Linear(2*width, width))
        self.norm = nn.LayerNorm(width)
        self.action_score = nn.Linear(width, width, bias=False)
        self.outcome_head = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, outcome_classes))
        self.value_head = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, evidence: Tensor, candidates: Tensor, *, state: Tensor | None = None,
                evidence_mask: Tensor | None = None, steps: int = 2) -> dict[str, Tensor]:
        if type(steps) is not int or not 1 <= steps <= 128:
            raise ValueError('explicit reference-kernel step budget is 1..128')
        if evidence.ndim != 3 or candidates.ndim != 3:
            raise ValueError('expected evidence [B,N,d], candidates [B,A,d]')
        b, n, d = evidence.shape
        if b < 1 or n < 1 or d != self.width or candidates.shape[0] != b or candidates.shape[1] < 1 or candidates.shape[2] != d:
            raise ValueError('invalid feature dimensions')
        if evidence.device != candidates.device or evidence.dtype != candidates.dtype:
            raise ValueError('feature dtype/device mismatch')
        if state is None:
            z = self.initial[None].expand(b, -1, -1)
        else:
            if state.shape != (b, self.slots, d) or state.device != evidence.device or state.dtype != evidence.dtype:
                raise ValueError('invalid state')
            z = state
        mask = evidence_mask
        if mask is not None:
            if mask.shape != (b, n) or mask.dtype != torch.bool or mask.device != evidence.device:
                raise ValueError('invalid evidence mask')
            if not bool(mask.any(-1).all()):
                raise ValueError('each example needs at least one visible evidence row')
        # Masked rows still have to contain valid finite numbers; this kernel
        # deliberately does not promise to repair corrupt/missing feature data.
        if not bool(torch.isfinite(evidence).all()) or not bool(torch.isfinite(candidates).all()) or not bool(torch.isfinite(z).all()):
            raise ValueError('nonfinite input feature/state')
        keys, values = self.k(evidence), self.v(evidence)
        for _ in range(steps):
            logits = torch.matmul(self.q(z), keys.transpose(-1, -2)) / math.sqrt(d)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, :], -torch.inf)
            context = logits.softmax(-1) @ values
            joined = torch.cat((z, context), -1)
            gate = self.gate(joined).sigmoid()
            z = self.norm((1-gate)*z + gate*self.proposal(joined))
        pooled = z.mean(1)
        scores = torch.einsum('bd,bad->ba', self.action_score(pooled), candidates) / math.sqrt(d)
        paired = torch.cat((pooled[:, None].expand(-1, candidates.shape[1], -1), candidates), -1)
        return {'state': z, 'action_logits': scores,
                'outcome_logits': self.outcome_head(paired),
                'estimated_progress': self.value_head(paired).squeeze(-1)}


class ResearchBridge(nn.Module):
    """Zero-output initial residual bridge. NOT automatically inserted into HF.

    Auxiliary losses on ResearchStateCore can train the core even while this
    output projection is zero. A nonzero bridge output alone is not competence.
    """
    def __init__(self, backbone_width: int, research_width: int):
        super().__init__()
        if backbone_width < 1 or research_width < 1:
            raise ValueError('positive widths required')
        self.width = research_width
        self.query = nn.Linear(backbone_width, research_width, bias=False)
        self.key = nn.Linear(research_width, research_width, bias=False)
        self.value = nn.Linear(research_width, research_width, bias=False)
        self.output = nn.Linear(research_width, backbone_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3 or state.ndim != 3 or hidden.shape[0] != state.shape[0]:
            raise ValueError('invalid hidden/state shapes')
        if state.shape[-1] != self.width or hidden.shape[-1] != self.query.in_features:
            raise ValueError('invalid feature widths')
        weights = (self.query(hidden) @ self.key(state).transpose(-1, -2) / math.sqrt(self.width)).softmax(-1)
        residual = self.output(weights @ self.value(state))
        return hidden + residual, residual
