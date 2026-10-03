"""Pinned research implementation on the admitted, reduced Qwen text fixture.

Research source modules are imported in full and hash checked. TextEntry retains
its documented framework/mask/cache scaffold; this is not True or real weights.
The optional nonzero bridge is explicit test initialization, after the unchanged
native installer has created its original zero-output bridge.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

from tools.text_model_entry import TextEntry

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "upstream" / "research_native"
PINS = {
    "experiments/ornith_problem_formation_v1/research_neural.py": "4cf397aa49380b85c7ff83def6fa52a49012480253ba65b6d7256b044fbb2a2e",
    "experiments/ornith_problem_formation_v1/attention_memory.py": "9ac11b6a53a037f3dbad8bc7ef156d72d00705da2b72cf87374c2a1eb54753cd",
    "experiments/ornith_problem_formation_v1/test_research_neural.py": "1ffdd18b8f495bff58d9a18045b591e3bdeffb4350052c74cba8ef8e1a6ae51a",
    "experiments/ornith_research_credit_v1/baseline/research_primitives.py": "b1b5b24c2190777cd1db8229a7cfaf0ba8191ea334b9f3a6a3d124dd0ce25ead",
}
MODULES = (
    "experiments.ornith_research_credit_v1.baseline.research_primitives",
    "experiments.ornith_problem_formation_v1.attention_memory",
    "experiments.ornith_problem_formation_v1.research_neural",
)


def verify_sources():
    manifest = json.loads((ROOT / "provenance" / "RESEARCH_NATIVE.json").read_text(encoding="utf-8"))
    if manifest["base_commit"] != "806a44767ad64445ff324e1ae41dca77acf23d6c" or set(manifest["files"]) != set(PINS):
        raise ValueError("RESEARCH_SOURCE_MANIFEST_MISMATCH")
    result = {}
    for relative, digest in PINS.items():
        raw = (NATIVE_ROOT / relative).read_bytes()
        entry = manifest["files"][relative]
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if (hashlib.sha256(raw).hexdigest() != digest or entry["sha256"] != digest
                or entry["bytes"] != len(raw) or entry["git_blob"] != blob):
            raise ValueError("RESEARCH_SOURCE_HASH_MISMATCH: " + relative)
        result[relative] = {"sha256": digest, "git_blob": blob, "bytes": len(raw)}
    return {"base_commit": manifest["base_commit"], "files": result}


def native_modules():
    verify_sources()
    location = str(NATIVE_ROOT)
    if location not in sys.path:
        sys.path.insert(0, location)
    loaded = []
    for name in MODULES:
        module = importlib.import_module(name)
        expected = (NATIVE_ROOT / (name.replace(".", "/") + ".py")).resolve()
        if Path(module.__file__).resolve() != expected:
            raise ValueError("RESEARCH_IMPORT_FROM_DIFFERENT_TREE: " + name)
        loaded.append(module)
    return tuple(loaded)


class ResearchLabModel(nn.Module):
    """Full reduced text-class path with the original research backbone shape."""

    def __init__(self, *, vocab_size=17, layer_types=None):
        super().__init__()
        kwargs = {} if layer_types is None else {"layer_types": tuple(layer_types)}
        self.model = TextEntry(dtype=torch.float32, vocab_size=vocab_size, **kwargs)
        self.config = self.model.config

    def forward(self, input_ids=None, *, use_cache=False, attention_mask=None, inputs_embeds=None):
        if use_cache is not False:
            raise ValueError("RESEARCH_FIXTURE_REQUIRES_FULL_PREFIX_NO_CACHE")
        return SimpleNamespace(logits=self.model(input_ids, attention_mask=attention_mask,
                                                inputs_embeds=inputs_embeds))


def build(*, seed=9028, width=4, slots=2, steps=1, vocab_size=17, layer=-1,
          outcome_classes=2, layer_types=None, nonzero_bridge=False):
    """Return (model, native runtime); base frozen, research trainable FP32 CPU.

    width is the research latent width. The admitted backbone has hidden_size=4.
    nonzero_bridge=True is for gradient reachability witnesses, not upstream
    installation semantics or a claim about pretrained initialization.
    """
    if type(seed) is not int or type(nonzero_bridge) is not bool:
        raise ValueError("EXPLICIT_INTEGER_SEED_AND_BOOLEAN_BRIDGE_REQUIRED")
    _, _, native = native_modules()
    # Import the real dependency before recording any candidate arithmetic.
    # No stub or alternative attention implementation is installed here.
    importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResearchLabModel(vocab_size=vocab_size, layer_types=layer_types)
        model.requires_grad_(False)
        model.eval()
        before = {id(p): (p.detach().clone(), p.requires_grad) for p in model.parameters()}
        runtime = native.install_research_neural(model, width=width, slots=slots,
                  layer=layer, steps=steps, outcome_classes=outcome_classes)
        for p in model.model.parameters():
            original, flag = before[id(p)]
            if p.requires_grad != flag or not torch.equal(p, original):
                raise ValueError("NATIVE_INSTALL_CHANGED_BASE")
        if torch.count_nonzero(runtime.components.bridge.output.weight).item() != 0:
            raise ValueError("NATIVE_INSTALL_NONZERO_BRIDGE")
        if nonzero_bridge:
            with torch.no_grad():
                bridge = runtime.components.bridge.output.weight
                # Deterministic random test data, not an implementation change.
                bridge.copy_(torch.randn_like(bridge) * 0.08)
            runtime.mark_updated()
    model.fixture_spec = {"seed": seed, "width": width, "slots": slots, "steps": steps,
        "vocab_size": vocab_size, "layer": runtime.layer, "outcome_classes": outcome_classes,
        "layer_types": list(model.config.layer_types), "nonzero_bridge": nonzero_bridge,
        "backbone_hidden_size": model.config.hidden_size, "base_frozen": True}
    return model, runtime


def identity(model):
    """Pure-data configuration/source identity; no process IDs or tensor pointers."""
    return {"schema": "xiaomai-research-fixture-v1", "research_sources": verify_sources(),
            "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "text_source_sha256": model.model.receipt["source_sha256"],
            "fixture_spec": dict(model.fixture_spec),
            "real_assets": False, "current_true_restored": False,
            "scaffolding": model.model.receipt["remaining_scaffolding"],
            "attention_workspace_scope": "Original research context imports installed transformers; reduced TextEntry uses its own pinned eager namespace. Short sequences use original eager, no tiling claim."}
