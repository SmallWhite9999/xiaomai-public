"""Evidence-reader tests; usable with stdlib unittest or the existing pytest CI."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_admission_reader", ROOT / "examples/read_admission_summary.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("20261013_r20", "20261014_multilayer_cpu"):
            shutil.copytree(ROOT / "evidence" / name, self.root / "evidence" / name)
        self.cpu = self.root / "evidence/20261014_multilayer_cpu"
        self.r20 = self.root / "evidence/20261013_r20"

    def edit(self, path, mutation, rehash=True):
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
        if rehash:
            # Adversarial records deliberately receive a valid new checksum.
            # Semantic checks must still reject the inconsistent numeric record.
            self.rehash(path)

    def rehash(self, path):
        provenance_path = path.parent / "SOURCE_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    def reject(self):
        with self.assertRaises(reader.EvidenceError):
            reader.read_evidence(self.root)

    def test_published_records_keep_scientific_failure_separate(self):
        result = reader.read_evidence(self.root)
        self.assertEqual(result["public_numeric_record_check"], "PASS")
        self.assertFalse(result["model_rerun"])
        self.assertFalse(result["full_tensor_recomparison"])
        cpu = result["multilayer_cpu"]
        self.assertEqual((cpu["cases"], cpu["case_pass"], cpu["case_fail"]), (14, 0, 14))
        self.assertEqual((cpu["roles"], cpu["failed_elements"]), (1066, 1776))
        self.assertEqual(cpu["scientific_multilayer_FP32_admission"], "FAIL")
        self.assertEqual(result["r20"]["mlp_compared_elements"], 151027712)

    def test_modified_bytes_rejected_before_arithmetic(self):
        self.edit(self.cpu / "cases.json", lambda x: x["cases"][0].update(bad_elements=0),
                  rehash=False)
        self.reject()

    def test_coordinate_value_tampering_rejected_with_new_hash(self):
        self.edit(self.cpu / "role_metrics.json",
                  lambda x: x["roles"][0]["worst_coordinate"].update(candidate=123.0))
        self.reject()

    def test_incomplete_roles_rejected_with_new_hash(self):
        self.edit(self.cpu / "role_metrics.json", lambda x: x["roles"].pop())
        self.reject()

    def test_duplicate_roles_rejected_with_new_hash(self):
        self.edit(self.cpu / "role_metrics.json", lambda x: x["roles"].append(x["roles"][0]))
        self.reject()

    def test_missing_case_rejected_with_new_hash(self):
        self.edit(self.cpu / "cases.json", lambda x: x["cases"].pop())
        self.reject()

    def test_case_count_tampering_rejected_with_new_hash(self):
        self.edit(self.cpu / "cases.json", lambda x: x["cases"][0].update(bad_elements=0))
        self.reject()

    def test_false_scientific_pass_rejected_with_new_hash(self):
        self.edit(self.cpu / "summary.json",
                  lambda x: x.update(scientific_multilayer_FP32_admission="PASS"))
        self.reject()

    def test_relaxed_threshold_rejected_with_new_hash(self):
        self.edit(self.cpu / "summary.json",
                  lambda x: x["numeric_contract"].update(atol=1.0))
        self.reject()

    def test_negative_controls_cannot_be_omitted(self):
        self.edit(self.cpu / "summary.json", lambda x: x.pop("negative_controls"))
        self.reject()

    def test_negative_control_accounting_rejected(self):
        self.edit(self.cpu / "summary.json",
                  lambda x: x["negative_controls"].update(fail=0))
        self.reject()

    def test_rewritten_control_success_cannot_replace_original_failure(self):
        self.edit(self.cpu / "summary.json",
                  lambda x: x["negative_controls"].update(
                      fail=0, **{"pass": 112, "cases_complete_pass": 14,
                                 "cases_with_failures": 0}))
        self.reject()

    def test_later_synthetic_patch_cannot_become_model_admission(self):
        self.edit(self.cpu / "summary.json",
                  lambda x: x["unapplied_control_fix"].update(new_model_admission=True))
        self.reject()

    def test_complete_layer_claim_rejected_with_new_hash(self):
        self.edit(self.r20 / "summary.json",
                  lambda x: x["mlp_layer0"].update(full_decoder_layer=True))
        self.reject()

    def test_missing_artifact_rejected(self):
        (self.cpu / "role_metrics.json").unlink()
        self.reject()

    def test_nonfinite_coordinate_rejected_with_new_hash(self):
        path = self.cpu / "role_metrics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["roles"][0]["worst_coordinate"]["candidate"] = float("nan")
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        self.rehash(path)
        self.reject()

    def test_boolean_count_rejected_with_new_hash(self):
        self.edit(self.cpu / "role_metrics.json",
                  lambda x: x["roles"][0].update(failed_elements=False))
        self.reject()

    def test_duplicate_json_key_rejected_with_new_hash(self):
        path = self.cpu / "summary.json"
        raw = path.read_text(encoding="utf-8")
        path.write_text('{"schema":"forged",' + raw[1:], encoding="utf-8")
        self.rehash(path)
        self.reject()


if __name__ == "__main__":
    unittest.main()
