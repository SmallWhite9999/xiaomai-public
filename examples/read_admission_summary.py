"""Check published numeric records with the standard library; never run a model.

The role records contain reported worst coordinates and counts, not full tensors.
Hashes detect changes relative to the accompanying provenance file. They do not
authenticate authorship if an attacker can replace the provenance file as well.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


class EvidenceError(ValueError):
    """A public record is incomplete, inconsistent, or unsupported."""


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise EvidenceError(f"non-finite JSON constant: {value}")


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {path.name}: {exc}") from exc


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value),
             f"{label}: finite number required")
    return value


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum,
             f"{label}: integer >= {minimum} required")
    return value


def _equal_number(actual, expected, label):
    _number(actual, label)
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-18),
             f"{label}: reported {actual!r} != recomputed {expected!r}")


def _hash_bundle(directory, filenames):
    provenance = _read_json(directory / "SOURCE_PROVENANCE.json")
    _require(provenance.get("schema") == "XIAOMAI_PUBLIC_WHITELIST_PROVENANCE_V1",
             "unsupported provenance schema")
    artifacts = provenance.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == set(filenames),
             "provenance artifact set is incomplete or unexpected")
    for filename in filenames:
        pin = artifacts[filename]
        _require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin),
                 f"invalid artifact SHA-256: {filename}")
        try:
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        except OSError as exc:
            raise EvidenceError(f"missing artifact: {filename}") from exc
        _require(digest == pin, f"artifact SHA-256 mismatch: {filename}")
    sources = provenance.get("source_snapshots")
    _require(isinstance(sources, list) and len(sources) > 0,
             "source snapshots are missing")
    for source in sources:
        _require(isinstance(source.get("sha256"), str) and
                 re.fullmatch(r"[0-9a-f]{64}", source["sha256"]),
                 "invalid source SHA-256")
    return {name: _read_json(directory / name) for name in filenames}


def _check_point(point, elements, atol, rtol, label):
    _require(isinstance(point, dict), f"{label}: worst coordinate is missing")
    for field in ("candidate", "reference", "abs_difference", "bound", "error_over_bound"):
        _number(point.get(field), f"{label}.{field}")
    index = point.get("index")
    _require(isinstance(index, list) and len(index) > 0,
             f"{label}: coordinate index missing")
    for coordinate in index:
        _integer(coordinate, f"{label}.index")
    flat = _integer(point.get("flat_index"), f"{label}.flat_index")
    _require(flat < elements, f"{label}: flat index outside reported element count")
    difference = abs(point["candidate"] - point["reference"])
    bound = atol + rtol * abs(point["reference"])
    _equal_number(point["abs_difference"], difference, f"{label}.abs_difference")
    _equal_number(point["bound"], bound, f"{label}.bound")
    _equal_number(point["error_over_bound"], difference / bound,
                  f"{label}.error_over_bound")
    return difference / bound


def validate_multilayer(summary, case_data, role_data):
    """Recheck reported coordinates and aggregate counts; no tensor recomputation."""
    _require(summary.get("schema") == "XIAOMAI_PUBLIC_MULTILAYER_CPU_SUMMARY_V1",
             "unsupported multilayer summary schema")
    _require(case_data.get("schema") == "XIAOMAI_PUBLIC_MULTILAYER_CPU_CASES_V1",
             "unsupported case schema")
    _require(role_data.get("schema") == "XIAOMAI_PUBLIC_MULTILAYER_CPU_ROLES_V1",
             "unsupported role schema")
    contract = summary["numeric_contract"]
    atol = _number(contract["atol"], "atol")
    rtol = _number(contract["rtol"], "rtol")
    _require(atol == 8e-6 and rtol == 5e-4, "unsupported or relaxed numeric contract")
    _require(summary["execution_context"]["device"] == "CPU" and
             summary["execution_context"]["synthetic_inputs"] is True,
             "unsupported execution scope")
    _require(summary["limitations"]["GPU_execution"] is False and
             summary["limitations"]["True_integration"] is False and
             summary["limitations"]["full_tensors_included"] is False,
             "unsupported model acceptance claim")
    cases = case_data["cases"]
    rows = role_data["roles"]
    _require(isinstance(cases, list) and isinstance(rows, list), "record arrays required")
    expected_ids = summary["case_ids"]
    _require(isinstance(expected_ids, list) and len(expected_ids) == 14 and
             len(set(expected_ids)) == 14, "14 distinct case IDs required")
    _require(len(cases) == len(expected_ids) and
             {c["case"] for c in cases} == set(expected_ids),
             "case coverage is incomplete or duplicated")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        case, role = row["case"], row["role"]
        _require(case in expected_ids and isinstance(role, str) and len(role) > 0,
                 "unexpected case or missing role")
        _require((case, role) not in seen, f"duplicate role: {case}/{role}")
        seen.add((case, role))
        total = _integer(row["elements"], "role.elements", 1)
        failed = _integer(row["failed_elements"], "role.failed_elements")
        _require(failed <= total, "failed element count exceeds role size")
        ratio = _check_point(row["worst_coordinate"], total, atol, rtol, f"{case}/{role}")
        _equal_number(row["worst_ratio"], ratio, "role.worst_ratio")
        _require(type(row["passed"]) is bool and
                 row["passed"] == (failed == 0) == (ratio <= 1),
                 "role verdict disagrees with count or worst coordinate")
        for field in ("near_zero_reference_elements", "signal_reference_elements_ge10atol"):
            _require(_integer(row[field], field) <= total, f"{field} exceeds role size")
        grouped[case].append(row)
    for case in cases:
        case_id = case["case"]
        case_rows = grouped[case_id]
        _require(len(case_rows) > 0, f"no roles for {case_id}")
        worst = max(case_rows, key=lambda row: row["worst_ratio"])
        for field, recomputed in (
            ("roles", len(case_rows)),
            ("bad_roles", sum(not r["passed"] for r in case_rows)),
            ("bad_elements", sum(r["failed_elements"] for r in case_rows)),
            ("total_elements", sum(r["elements"] for r in case_rows)),
        ):
            _require(_integer(case[field], f"case.{field}") == recomputed,
                     f"case {field} mismatch: {case_id}")
        _equal_number(case["worst_ratio"], worst["worst_ratio"], "case.worst_ratio")
        _require(case["worst_role"] == worst["role"] and
                 case["worst_point"] == worst["worst_coordinate"],
                 f"case worst role or coordinate mismatch: {case_id}")
    totals = summary["reported_totals"]
    recomputed = {
        "cases": len(cases),
        "case_pass": sum(c["bad_elements"] == 0 for c in cases),
        "case_fail": sum(c["bad_elements"] > 0 for c in cases),
        "roles": len(rows),
        "failed_roles": sum(not r["passed"] for r in rows),
        "elements": sum(r["elements"] for r in rows),
        "failed_elements": sum(r["failed_elements"] for r in rows),
    }
    for field, value in recomputed.items():
        _require(_integer(totals[field], f"totals.{field}") == value,
                 f"summary aggregate mismatch: {field}")
    worst_ratio = max(r["worst_ratio"] for r in rows)
    _equal_number(totals["max_error_over_bound"], worst_ratio, "summary.max_error_over_bound")
    expected_admission = "FAIL" if recomputed["case_fail"] else "PASS"
    _require(summary["scientific_multilayer_FP32_admission"] == expected_admission,
             "scientific admission disagrees with case verdicts")
    controls = summary["negative_controls"]
    for field in ("pass", "fail", "total", "cases_complete_pass", "cases_with_failures"):
        _integer(controls[field], f"negative_controls.{field}")
    _require(controls["pass"] + controls["fail"] == controls["total"] == 112 and
             controls["cases_complete_pass"] + controls["cases_with_failures"] == len(cases) and
             (controls["fail"] == 0) == (controls["cases_with_failures"] == 0),
             "negative-control accounting mismatch")
    _require(summary["G_comparison"]["raw_byte_equivalence_pass"] == len(cases) and
             summary["G_comparison"]["cases"] == len(cases) and
             summary["G_comparison"]["full_protocol_verdict"] == "FAIL" and
             controls["fail"] > 0,
             "G bit equivalence and full protocol verdict must remain distinct")
    _require(summary["unapplied_control_fix"]["applied_to_original_run"] is False and
             summary["unapplied_control_fix"]["new_model_admission"] is False,
             "original failure cannot be replaced by a later synthetic fix")
    return {**recomputed, "max_error_over_bound": worst_ratio,
            "scientific_multilayer_FP32_admission": expected_admission}


def validate_r20(summary):
    _require(summary.get("schema") == "XIAOMAI_PUBLIC_R20_EVIDENCE_V1",
             "unsupported R20 summary schema")
    mlp = summary["mlp_layer0"]
    _require(mlp["status"] == "PASS_FP32_REAL_WEIGHT_ARTIFICIAL_INPUT_FORWARD_AND_EXPLICIT_VJP_CUDA",
             "unsupported MLP status")
    _require(mlp["weight_numel"] == 150994944 and mlp["compared_elements"] == 151027712,
             "unexpected bounded MLP comparison size")
    _require(mlp["compute_dtype"] == "FP32" and mlp["device"] == "CUDA" and
             mlp["BF16_weights_cast_to_FP32_reference"] is True and
             mlp["candidate_autograd_graph"] is False,
             "unsupported MLP precision claim")
    _require(mlp["artificial_input_and_dY_shape"] == [4, 4096],
             "unexpected artificial MLP input domain")
    atol, rtol = mlp["fixed_atol"], mlp["fixed_rtol"]
    _require(atol == 8e-6 and rtol == 5e-4, "unsupported R20 tolerance")
    error = _number(mlp["max_abs_error"], "mlp.max_abs_error")
    _require(0 <= error <= atol, "reported maximum error exceeds absolute tolerance")
    for field in ("production_BF16", "integrated_into_formal_model", "full_decoder_layer"):
        _require(mlp[field] is False, f"unverified MLP acceptance claim: {field}")
    compact = summary["selected_research_gradients"]
    _require(compact["status"] == "PASS_SELECTED_17_RESEARCH_GRADIENT_COMPARISON" and
             compact["fixed_atol"] == 8e-6 and compact["fixed_rtol"] == 5e-4,
             "unsupported selected-gradient result or tolerance")
    _require(compact["admitted_gradients"] == 17 and compact["comparison_rows"] == 18 and
             compact["comparison_elements"] == 143777,
             "unexpected selected-gradient comparison size")
    for field in ("full_9B_or_complete_layer_acceptance", "full_trace_off_on_pair_acceptance",
                  "new_saved_checkpoint"):
        _require(compact[field] is False, f"unverified compact acceptance claim: {field}")
    _require(summary["complete_decoder_layer_acceptance"] == {"accepted": 0, "denominator": 32},
             "complete layer coverage was not established")
    for field in ("model_rerun_by_public_reader", "raw_tensor_data_included",
                  "full_model_training_acceptance", "capability_or_autonomy_acceptance"):
        _require(summary["limitations"][field] is False, f"unsupported R20 claim: {field}")
    return {"mlp_compared_elements": mlp["compared_elements"],
            "mlp_max_abs_error": error,
            "selected_gradients": compact["admitted_gradients"],
            "selected_comparison_rows": compact["comparison_rows"],
            "selected_comparison_elements": compact["comparison_elements"]}


def read_evidence(repo_root):
    root = Path(repo_root)
    try:
        r20 = _hash_bundle(root / "evidence/20261013_r20", ("summary.json",))
        cpu = _hash_bundle(root / "evidence/20261014_multilayer_cpu",
                           ("summary.json", "cases.json", "role_metrics.json"))
        result = {
            "public_numeric_record_check": "PASS",
            "model_rerun": False,
            "full_tensor_recomparison": False,
            "r20": validate_r20(r20["summary.json"]),
            "multilayer_cpu": validate_multilayer(cpu["summary.json"], cpu["cases.json"],
                                                  cpu["role_metrics.json"]),
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise EvidenceError(f"incomplete or malformed evidence: {exc}") from exc
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = read_evidence(args.repo_root)
    except EvidenceError as exc:
        parser.exit(1, f"PUBLIC_EVIDENCE_REJECTED: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
