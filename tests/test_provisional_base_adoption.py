import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_unknown_campaign import sha256_file, sha256_obj, sha256_tree, validate_panel_dispatch
from scripts.run_unknown_supervisor import materialize_provisional_base_adoption_queue


def _artifact(tmp_path, name):
    path = tmp_path / name
    path.write_text(json.dumps({"name": name}), encoding="utf-8")
    return str(path), sha256_file(path)


def test_adoption_dispatch_requires_machine_readable_receipt_projection_and_rule_c(monkeypatch, tmp_path):
    source_config_snapshot_sha256 = "training-time-source-config-sha"
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"source_config_snapshot_sha256": source_config_snapshot_sha256}), encoding="utf-8")
    receipt, receipt_sha = str(receipt_path), sha256_file(receipt_path)
    selector, selector_sha = _artifact(tmp_path, "selector.json")
    offline, offline_sha = _artifact(tmp_path, "offline.json")
    base = {"id": "base", "tier": 3, "step": "base", "seed": 42}
    lr = {"id": "lr008", "tier": 3, "step": "lr008", "seed": 42}
    artifact = tmp_path / "base-artifact"; artifact.mkdir(); (artifact / "result.json").write_text("{}")
    projection_payload = {**base, "run_id": "canonical-base", "artifact_path": str(artifact),
                          "artifact_sha256": sha256_tree(artifact), "source_config_snapshot_sha256": source_config_snapshot_sha256,
                          "rule_c_artifacts": [{"selection_snapshot_path": selector, "selection_snapshot_sha256": selector_sha,
                                                "offline_path": offline, "offline_sha256": offline_sha}]}
    projection_path = tmp_path / "projection.json"; projection_path.write_text(json.dumps(projection_payload), encoding="utf-8")
    projection, projection_sha = str(projection_path), sha256_file(projection_path)
    adoption = {"base": base, "source_config_snapshot_sha256": source_config_snapshot_sha256,
                "receipt": {"path": receipt, "sha256": receipt_sha},
                "projection": {"path": projection, "sha256": projection_sha}}
    evidence = {"action": "provisional_base_adoption_lr008", "proposed_queue": [lr],
                "provisional_base_adoption": adoption,
                "completed": [], "failed": [], "config_snapshot_sha256": "current-adoption-config-sha",
                "binding": {"config_snapshot_sha256": "current-adoption-config-sha", "ordered_queue_sha256": sha256_obj([lr])}}
    response = {"provisional_base_adoption": {"decision": "adopt", "receipt_path": receipt,
                "receipt_sha256": receipt_sha, "projection_path": projection, "projection_sha256": projection_sha}}
    monkeypatch.setattr("scripts.run_unknown_campaign.validate_panel_bundle", lambda *_: (True, "ok", {"response": response}))
    monkeypatch.setattr("scripts.run_unknown_campaign.read_json", lambda path: evidence if Path(path).name == "evidence_packet.json" else json.loads(Path(path).read_text()))
    monkeypatch.setattr("scripts.run_unknown_campaign.recompute_contract", lambda *_: {"ordered_queue_sha256": sha256_obj([lr])})
    cfg = {"outer_loop": {"provisional_base_adoption": adoption, "provisional_base_adoption_lr008_queue": [lr]}}
    assert validate_panel_dispatch(tmp_path / "r3.json", None, lr, "provisional_base_adoption_lr008", "current-adoption-config-sha", cfg)[0]
    evidence["completed"] = [base]
    assert validate_panel_dispatch(tmp_path / "r3.json", None, lr, "provisional_base_adoption_lr008", "current-adoption-config-sha", cfg)[1] == "panel_provisional_base_adoption_base_state_preexisting"
    evidence["completed"] = []
    response["provisional_base_adoption"]["decision"] = "reject"
    assert validate_panel_dispatch(tmp_path / "r3.json", None, lr, "provisional_base_adoption_lr008", "current-adoption-config-sha", cfg)[1] == "panel_provisional_base_adoption_not_adopted"
    response["provisional_base_adoption"]["decision"] = "adopt"
    projection_payload["source_config_snapshot_sha256"] = "tampered-source-config-sha"
    projection_path.write_text(json.dumps(projection_payload), encoding="utf-8")
    adoption["projection"]["sha256"] = sha256_file(projection_path)
    response["provisional_base_adoption"]["projection_sha256"] = adoption["projection"]["sha256"]
    assert validate_panel_dispatch(tmp_path / "r3.json", None, lr, "provisional_base_adoption_lr008", "current-adoption-config-sha", cfg)[1] == "panel_provisional_base_adoption_projection_source_config_mismatch"
    projection_payload["source_config_snapshot_sha256"] = source_config_snapshot_sha256
    projection_payload["rule_c_artifacts"] = []  # no consensus / no valid Rule-C selector+offline pair
    projection_path.write_text(json.dumps(projection_payload), encoding="utf-8")
    adoption["projection"]["sha256"] = sha256_file(projection_path)
    response["provisional_base_adoption"]["projection_sha256"] = adoption["projection"]["sha256"]
    assert validate_panel_dispatch(tmp_path / "r3.json", None, lr, "provisional_base_adoption_lr008", "cfg", cfg)[1] == "panel_provisional_base_adoption_rule_c_artifact_invalid"


def test_adoption_materializes_only_lr008_at_index_zero(monkeypatch, tmp_path):
    base = {"id": "base", "tier": 3, "step": "base", "seed": 42}
    lr = {"id": "lr008", "tier": 3, "step": "lr008", "seed": 42}
    artifact = tmp_path / "base-artifact"; artifact.mkdir(); (artifact / "result.json").write_text("{}")
    projection_payload = {**base, "run_id": "canonical-base", "artifact_path": str(artifact),
                          "artifact_sha256": sha256_tree(artifact), "source_config_snapshot_sha256": "training-time-source-config-sha",
                          "rule_c_artifacts": []}
    projection_path = tmp_path / "projection.json"; projection_path.write_text(json.dumps(projection_payload), encoding="utf-8")
    projection = {"path": str(projection_path), "sha256": sha256_file(projection_path)}
    adoption = {"base": base, "source_config_snapshot_sha256": "training-time-source-config-sha",
                "receipt": {"path": "receipt", "sha256": "receipt-sha"}, "projection": projection}
    state = {"outer_loop": {"provisional_base_adoption": adoption, "provisional_base_adoption_lr008_queue": [lr]},
             "completed": [{"id": "nonofficial"}], "failed": [], "queue": [], "panel": {"approved": True, "action": "provisional_base_adoption_lr008",
             "r3_artifact_path": str(tmp_path / "r3.json"), "evidence_packet_sha": "evidence"}}
    monkeypatch.setattr("scripts.run_unknown_supervisor.validate_panel_dispatch", lambda *_: (True, "ok"))
    config = tmp_path / "config.json"; config.write_text("{}", encoding="utf-8")
    assert materialize_provisional_base_adoption_queue(state, {"outer_loop": state["outer_loop"]}, config)
    assert state["queue"] == [{**lr, "attempts": 0, "approved_queue_index": 0,
                               "panel_r3_artifact": state["panel"]["r3_artifact_path"], "evidence_packet_sha": "evidence"}]
    assert state["adopted_base_projections"] == [projection]
    assert state["completed"][-1] == projection_payload
