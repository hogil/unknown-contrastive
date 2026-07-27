import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.materialize_panel_round import finalize_panel_round, prepare_panel_prompt
from scripts.run_unknown_campaign import sha256_file, sha256_obj, validate_panel_bundle

CONTRACT = {"A": ("gpt-5.6-sol", "max"), "B": ("gpt-5.6-terra", "ultra"), "C": ("gpt-5.6-sol", "ultra")}

def _write(path, value): path.write_text(json.dumps(value), encoding="utf-8")
def _time(minute): return f"2026-07-26T00:{minute:02d}:00+00:00"
def _prepare(evidence, panel, task, judge, round_name, minute):
    model, effort = CONTRACT[judge]
    return prepare_panel_prompt(evidence_packet=evidence, panel_dir=panel, task_file=task, judge=judge, round_name=round_name, model=model, reasoning_effort=effort, prepared_at=_time(minute))
def _final(panel, prompt, raw, judge, minute):
    model, effort = CONTRACT[judge]
    return finalize_panel_round(panel_dir=panel, prompt_path=prompt, prompt_sha256=sha256_file(prompt), agent_id=f"agent-{judge}", task_name=f"/panel/{judge}", model=model, reasoning_effort=effort, started_at=_time(minute), completed_at=_time(minute + 1), timestamp=_time(minute + 2), raw_response_path=raw)

def _bundle(tmp_path):
    evidence, task, panel, raws = tmp_path/"evidence.json", tmp_path/"task.md", tmp_path/"panel", tmp_path/"raw"; raws.mkdir(); task.write_text("Review this exact evidence.", encoding="utf-8")
    _write(evidence, {"binding": {"config_snapshot_sha256":"c", "source_snapshot_sha256":"s", "data_snapshot_sha256":"d", "backbone_snapshot_sha256":"b", "env_snapshot_sha256":"e", "ordered_queue_sha256":"q"}})
    for offset, judge in enumerate(CONTRACT):
        prompt = _prepare(evidence, panel, task, judge, "R1", offset)
        raw = raws/f"R1_{judge}.json"; _write(raw, {"real": f"R1-{judge}"}); _final(panel, prompt, raw, judge, 10 + offset * 2)
    for offset, judge in enumerate(CONTRACT):
        prompt = _prepare(evidence, panel, task, judge, "R2", 20 + offset)
        raw = raws/f"R2_{judge}.json"; _write(raw, {"real": f"R2-{judge}"}); _final(panel, prompt, raw, judge, 30 + offset * 2)
    prompt = _prepare(evidence, panel, task, "C", "R3", 40); raw = raws/"R3_C.json"; _write(raw, {"conclusion":"approve", "critical_objections":[], "minority_positions":[], "near_audit_adjudications":[]})
    return evidence, panel, _final(panel, prompt, raw, "C", 50)

def test_two_phase_complete_bundle_validates_and_prompts_bind_real_inputs(tmp_path):
    evidence, panel, r3 = _bundle(tmp_path)
    assert validate_panel_bundle(r3, sha256_obj(json.loads(evidence.read_text())))[0]
    r1 = json.loads((panel/"prompt_R1_A.json").read_text()); r2 = json.loads((panel/"prompt_R2_A.json").read_text())
    assert "peer_artifacts" not in r1 and "r1_response_shas" not in r1
    assert set(r2["peer_artifacts"]) == {"A", "B", "C"}
    assert all(Path(item["artifact_path"]).is_absolute() and item["artifact_sha256"] for item in r2["peer_artifacts"].values())

def test_tampered_prompt_and_missing_peers_fail_closed(tmp_path):
    evidence, panel, r3 = _bundle(tmp_path)
    prompt = panel/"prompt_R1_A.json"; original_sha = sha256_file(prompt); value = json.loads(prompt.read_text()); value["task_instructions"] = "tampered"; _write(prompt, value)
    raw = tmp_path/"raw.json"; _write(raw, {"real":"no"})
    model, effort = CONTRACT["A"]
    with pytest.raises(ValueError, match="prompt path/SHA mismatch"):
        finalize_panel_round(panel_dir=panel, prompt_path=prompt, prompt_sha256=original_sha, agent_id="agent-A", task_name="/panel/A", model=model, reasoning_effort=effort, started_at=_time(55), completed_at=_time(56), timestamp=_time(57), raw_response_path=raw)
    task = tmp_path/"task.md"; task.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required R1 peer"): _prepare(evidence, tmp_path/"fresh", task, "A", "R2", 1)

def test_whitespace_only_evidence_tamper_after_prepare_fails_finalize(tmp_path):
    evidence, task, panel, raw_dir = tmp_path/"evidence.json", tmp_path/"task.md", tmp_path/"panel", tmp_path/"raw"
    raw_dir.mkdir(); task.write_text("Review this exact evidence.", encoding="utf-8")
    _write(evidence, {"binding": {"config_snapshot_sha256":"c", "source_snapshot_sha256":"s", "data_snapshot_sha256":"d", "backbone_snapshot_sha256":"b", "env_snapshot_sha256":"e", "ordered_queue_sha256":"q"}})
    prompt = _prepare(evidence, panel, task, "A", "R1", 0)
    panel_evidence = panel / "evidence_packet.json"
    panel_evidence.write_text(json.dumps(json.loads(panel_evidence.read_text()), indent=4, sort_keys=True), encoding="utf-8")
    raw = raw_dir / "R1_A.json"; _write(raw, {"real": "R1-A"})
    with pytest.raises(ValueError, match="prompt evidence file SHA invalid"):
        _final(panel, prompt, raw, "A", 10)
