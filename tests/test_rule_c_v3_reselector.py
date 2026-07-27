import copy
import json
from pathlib import Path

import pytest

from scripts import run_rule_c_v3_reselector as v3
from scripts import run_rule_c_offline as offline


SOURCE = Path(r"D:\project\unknown-contrastive\runs\campaign_state\rule_c\strict_novel_base_seed42_rulec_v2_diag_260726_215920\selection_snapshot.json")
BINDING = Path(r"D:\project\unknown-contrastive\runs\campaign_state\panels\unknown-f2c1417e26399999\r3_C.json")
CHECKPOINT_DIR = Path(r"D:\project\unknown-contrastive\runs\may_repro\abl_provisional_strict_base_s42_w8_user_direct_B4_260726_192011\checkpoints")


def source_snapshot():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def test_frozen_v2_metrics_select_epoch_14_with_exact_raw_minimax():
    decision = v3.reselect(source_snapshot())
    assert decision["failures"] == []
    assert decision["selected"]["epoch"] == 14
    assert [row["epoch"] for row in decision["candidate_scores"]] == [8, 14]
    assert decision["candidate_scores"][1]["selection_key"] == [46.15, 29.656666666666666, 14]


def test_linear_p75_inclusive_gate_and_two_dial_fallback_are_not_relaxed():
    assert v3.linear_p75([1, 2, 3, 9]) == 4.5
    snap = source_snapshot()
    snap["per_dial"][0]["metrics"]["ep14"]["k"] = 16
    assert 14 in v3.reselect(snap)["per_dial"][0]["retained_epochs"]
    for row in snap["per_dial"][2]["metrics"].values():
        if isinstance(row, dict) and "k" in row:
            row["stability"] = 0
    decision = v3.reselect(snap)
    assert decision["selected"] is None
    assert "empty_three_dial_intersection" in decision["failures"]


@pytest.mark.parametrize("mutate, expected", [
    (lambda s: s.__setitem__("pool_sha256", "wrong"), "manifest_sha256_mismatch"),
    (lambda s: s["per_dial"][1]["metrics"]["ep01"].__setitem__("pre_reassign_noise", float("nan")), "dial2:nonfinite_metric:ep01"),
    (lambda s: s["per_dial"][1]["metrics"]["ep01"].pop("coherence"), "dial2:missing_metric_fields:ep01:coherence"),
    (lambda s: s["per_dial"][2].__setitem__("dial", {"ratio": .02}), "per_dial_contract_mismatch"),
])
def test_invariant_and_schema_mismatches_fail_closed(mutate, expected):
    snap = source_snapshot()
    mutate(snap)
    decision = v3.reselect(snap)
    assert expected in decision["failures"]
    assert decision["selected"] is None


def test_immutable_cli_records_raw_source_and_r3_bindings(tmp_path, capsys):
    out = tmp_path / "selection_snapshot_v3.json"
    seal = tmp_path / "epoch_seal.json"
    v3.main(["--source-v2-snapshot", str(SOURCE), "--r3-binding", str(BINDING), "--checkpoint-dir", str(CHECKPOINT_DIR), "--out", str(out), "--seal-out", str(seal)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["selected_epoch"] == 14 and saved["labels_used"] is False
    assert saved["source_v2_snapshot_sha256"] == v3.sha256_file(SOURCE)
    assert saved["r3_binding_sha256"] == v3.sha256_file(BINDING)
    assert saved["r3_evidence_packet_sha256"] == "f2c1417e263999997643c52412a3dd1db4cd35b1e1c78471d14f206736f607db"
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection_valid"] is True and printed["epoch_seal_sha256"] == v3.sha256_file(seal)
    sealed = json.loads(seal.read_text(encoding="utf-8"))
    assert sealed["selection_snapshot_sha256"] == v3.sha256_file(out) and sealed["selected_epoch"] == 14
    with pytest.raises(FileExistsError, match="immutable output"):
        v3.main(["--source-v2-snapshot", str(SOURCE), "--r3-binding", str(BINDING), "--checkpoint-dir", str(CHECKPOINT_DIR), "--out", str(out), "--seal-out", str(seal)])


def test_offline_boundary_rejects_v2_unsealed_and_tampered_bindings_before_labels(tmp_path):
    out, seal = tmp_path / "v3.json", tmp_path / "seal.json"
    v3.main(["--source-v2-snapshot", str(SOURCE), "--r3-binding", str(BINDING), "--checkpoint-dir", str(CHECKPOINT_DIR), "--out", str(out), "--seal-out", str(seal)])
    offline.validate_v3_before_labels(out, v3.sha256_file(out), seal, v3.sha256_file(seal))
    trap = tmp_path / "labeled-trap.json"; trap.write_text("not-json", encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid or legacy"):
        offline.main(["--selection-snapshot", str(SOURCE), "--expected-selection-sha256", v3.sha256_file(SOURCE), "--epoch-seal", str(seal), "--expected-epoch-seal-sha256", v3.sha256_file(seal), "--labeled-pool", str(trap), "--out", str(tmp_path / "offline.json")])
    seal.write_text("tampered", encoding="utf-8")
    with pytest.raises(SystemExit, match="epoch seal missing or hash mismatch"):
        offline.main(["--selection-snapshot", str(out), "--expected-selection-sha256", v3.sha256_file(out), "--epoch-seal", str(seal), "--expected-epoch-seal-sha256", "wrong", "--labeled-pool", str(trap), "--out", str(tmp_path / "offline.json")])


@pytest.mark.parametrize("section, key", [("source_bundle", "bundle_path"), ("source_bundle", "npz_path"), ("selected_checkpoint", "selected_checkpoint_path")])
def test_offline_rejects_tampered_bound_cache_or_checkpoint_before_labels(tmp_path, section, key):
    out, seal = tmp_path / "v3.json", tmp_path / "seal.json"
    v3.main(["--source-v2-snapshot", str(SOURCE), "--r3-binding", str(BINDING), "--checkpoint-dir", str(CHECKPOINT_DIR), "--out", str(out), "--seal-out", str(seal)])
    payload = json.loads(out.read_text(encoding="utf-8")); payload[section][key] = str(tmp_path / "missing")
    out.write_text(json.dumps(payload), encoding="utf-8")
    seal_payload = json.loads(seal.read_text(encoding="utf-8")); seal_payload["selection_snapshot_sha256"] = v3.sha256_file(out)
    seal.write_text(json.dumps(seal_payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="bound cache or checkpoint"):
        offline.validate_v3_before_labels(out, v3.sha256_file(out), seal, v3.sha256_file(seal))
