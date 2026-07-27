import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_unknown_campaign import sha256_file, sha256_obj, validate_near_audit_adjudications, validate_panel_bundle


CONTRACT = {"A": ("gpt-5.6-sol", "max"), "B": ("gpt-5.6-terra", "ultra"), "C": ("gpt-5.6-sol", "ultra")}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bundle(tmp_path):
    evidence = {"binding": {"config_snapshot_sha256": "c", "source_snapshot_sha256": "s",
                "data_snapshot_sha256": "d", "backbone_snapshot_sha256": "b",
                "env_snapshot_sha256": "e", "ordered_queue_sha256": "q"}}
    _write(tmp_path / "evidence_packet.json", evidence)
    evidence_sha, binding_sha = sha256_obj(evidence), sha256_obj(evidence["binding"])
    evidence_file_sha = sha256_file(tmp_path / "evidence_packet.json")
    panel_id = f"unknown-{evidence_sha[:16]}"
    r1, r2 = {}, {}
    def materialize(judge, round_name, response_shas=None, started="2026-07-26T00:00:00+00:00"):
        prompt = {"panel_id": panel_id, "judge": judge, "round": round_name,
                  "evidence_packet_sha": evidence_sha, "evidence_packet_file_sha256": evidence_file_sha}
        if round_name == "R2": prompt["r1_response_shas"] = response_shas
        if round_name == "R3": prompt["r2_response_shas"] = response_shas
        prompt_path = tmp_path / f"prompt_{round_name}_{judge}.json"; _write(prompt_path, prompt)
        prompt_sha = sha256_file(prompt_path)
        model, effort = CONTRACT[judge]
        receipt = {"agent_id": f"agent-{judge}", "task_name": f"/{judge}/{round_name}", "model": model,
                   "reasoning_effort": effort, "judge": judge, "round": round_name,
                   "prompt_sha256": prompt_sha, "started_at": started,
                   "completed_at": started}
        receipt_path = tmp_path / f"receipt_{round_name}_{judge}.json"; _write(receipt_path, receipt)
        return str(prompt_path), prompt_sha, str(receipt_path), sha256_file(receipt_path)
    for judge, (model, effort) in CONTRACT.items():
        response = {"judge": judge, "round": "R1"}
        prompt_path,prompt_sha,receipt_path,receipt_sha=materialize(judge,"R1",started="2026-07-26T00:00:00+00:00")
        p = {"panel_id": panel_id, "judge": judge, "round": "R1", "model": model,
             "reasoning_effort": effort, "prompt_sha": prompt_sha, "prompt_path":prompt_path,"prompt_sha256":prompt_sha,"receipt_path":receipt_path,"receipt_sha256":receipt_sha,"evidence_packet_sha": evidence_sha,"evidence_packet_file_sha256": evidence_file_sha,
             "binding_sha256": binding_sha, "timestamp": "2026-07-26T00:01:00+00:00", "response": response,
             "response_sha": sha256_obj(response)}
        r1[judge] = p["response_sha"]
        _write(tmp_path / f"r1_{judge}.json", p)
    for judge, (model, effort) in CONTRACT.items():
        response = {"judge": judge, "round": "R2"}
        prompt_path,prompt_sha,receipt_path,receipt_sha=materialize(judge,"R2",r1,"2026-07-26T00:02:00+00:00")
        p = {"panel_id": panel_id, "judge": judge, "round": "R2", "model": model,
             "reasoning_effort": effort, "prompt_sha": prompt_sha,"prompt_path":prompt_path,"prompt_sha256":prompt_sha,"receipt_path":receipt_path,"receipt_sha256":receipt_sha,"evidence_packet_sha": evidence_sha,"evidence_packet_file_sha256": evidence_file_sha,
             "binding_sha256": binding_sha, "timestamp": "2026-07-26T00:03:00+00:00", "response": response,
             "response_sha": sha256_obj(response), "r1_response_shas": r1}
        r2[judge] = p["response_sha"]
        _write(tmp_path / f"r2_{judge}.json", p)
    response = {"chair": "C"}
    prompt_path,prompt_sha,receipt_path,receipt_sha=materialize("C","R3",r2,"2026-07-26T00:04:00+00:00")
    r3 = {"panel_id": panel_id, "judge": "C", "round": "R3", "model": "gpt-5.6-sol",
          "reasoning_effort": "ultra", "prompt_sha": prompt_sha,"prompt_path":prompt_path,"prompt_sha256":prompt_sha,"receipt_path":receipt_path,"receipt_sha256":receipt_sha,"evidence_packet_sha": evidence_sha,"evidence_packet_file_sha256": evidence_file_sha,
          "binding_sha256": binding_sha, "timestamp": "2026-07-26T00:05:00+00:00", "response": response,
          "response_sha": sha256_obj(response), "r2_response_shas": r2, "conclusion": "approve"}
    _write(tmp_path / "r3_C.json", r3)
    return evidence_sha, tmp_path / "r3_C.json"


def test_complete_exact_bundle_is_required(tmp_path):
    evidence_sha, r3 = _bundle(tmp_path)
    assert validate_panel_bundle(r3, evidence_sha)[0]


def test_r3_only_or_wrong_b_contract_fails_closed(tmp_path):
    evidence_sha, r3 = _bundle(tmp_path)
    (tmp_path / "r1_A.json").unlink()
    assert validate_panel_bundle(r3, evidence_sha)[0] is False
    evidence_sha, r3 = _bundle(tmp_path)
    p = json.loads((tmp_path / "r1_B.json").read_text())
    p["model"] = "gpt-5.6-sol"
    _write(tmp_path / "r1_B.json", p)
    assert validate_panel_bundle(r3, evidence_sha)[0] is False


def test_tampered_cross_round_hash_fails_closed(tmp_path):
    evidence_sha, r3 = _bundle(tmp_path)
    p = json.loads((tmp_path / "r3_C.json").read_text())
    p["r2_response_shas"]["A"] = "tampered"
    _write(tmp_path / "r3_C.json", p)
    assert validate_panel_bundle(r3, evidence_sha)[0] is False


def test_whitespace_only_evidence_tamper_after_full_r3_fails_closed(tmp_path):
    evidence_sha, r3 = _bundle(tmp_path)
    evidence_path = tmp_path / "evidence_packet.json"
    evidence_path.write_text(json.dumps(json.loads(evidence_path.read_text()), indent=4, sort_keys=True), encoding="utf-8")
    assert validate_panel_bundle(r3, evidence_sha)[0] is False


def test_blind_prompt_receipt_and_distinct_response_guards_fail_closed(tmp_path):
    evidence_sha, r3 = _bundle(tmp_path)
    prompt_path = tmp_path / "prompt_R1_A.json"
    prompt = json.loads(prompt_path.read_text())
    prompt["peer_response_shas"] = {"B": "forbidden"}
    _write(prompt_path, prompt)
    assert validate_panel_bundle(r3, evidence_sha)[0] is False


def test_review_required_near_audit_needs_explicit_r3_closure(tmp_path):
    audit = tmp_path / "audit.json"
    report = {
        "near": {"candidate_pair_count": 1, "examples": [{"left": "a", "right": "b"}]},
        "provenance": {
            "cross_split_block_overlap_count": 2,
            "cross_split_block_overlap_examples": ["LOT_A", "LOT_B"],
        },
    }
    _write(audit, report)
    evidence = {"binding": {"recompute_contract": {"split_overlap_audits": [{
        "artifact": str(audit), "artifact_sha256": sha256_file(audit), "review_required": True}]}}}
    ok, reason = validate_near_audit_adjudications({}, evidence)
    assert not ok and reason == "panel_near_audit_adjudication_missing"
    closure = {"near_audit_adjudications": [{"audit_path": str(audit), "audit_sha256": sha256_file(audit),
        "near_candidate_pair_count": 1, "near_candidate_examples_sha256": sha256_obj(report["near"]["examples"]),
        "verdict": "approve_no_material_leakage", "rationale": "Verified distinct source lots."}]}
    ok, reason = validate_near_audit_adjudications(closure, evidence)
    assert not ok and reason == "panel_block_overlap_adjudication_invalid"
    closure["near_audit_adjudications"][0].update({
        "cross_split_block_overlap_count": 2,
        "cross_split_block_overlap_examples_sha256": sha256_obj(
            report["provenance"]["cross_split_block_overlap_examples"]),
        "block_verdict": "approve_no_material_source_leakage",
        "block_rationale": "Filename-token overlap was reviewed and is not source leakage.",
    })
    assert validate_near_audit_adjudications(closure, evidence) == (True, "ok")
    closure["near_audit_adjudications"][0]["verdict"] = "approve"
    assert validate_near_audit_adjudications(closure, evidence)[0] is False


def test_zero_block_overlap_keeps_existing_near_only_contract(tmp_path):
    audit = tmp_path / "audit_zero_blocks.json"
    report = {
        "near": {"candidate_pair_count": 0, "examples": []},
        "provenance": {"cross_split_block_overlap_count": 0},
    }
    _write(audit, report)
    evidence = {"binding": {"recompute_contract": {"split_overlap_audits": [{
        "artifact": str(audit), "artifact_sha256": sha256_file(audit), "review_required": True,
    }]}}}
    closure = {"near_audit_adjudications": [{
        "audit_path": str(audit),
        "audit_sha256": sha256_file(audit),
        "near_candidate_pair_count": 0,
        "near_candidate_examples_sha256": sha256_obj([]),
        "verdict": "approve_no_material_leakage",
        "rationale": "No near candidates were found.",
    }]}
    assert validate_near_audit_adjudications(closure, evidence) == (True, "ok")
    evidence_sha, r3 = _bundle(tmp_path)
    a = json.loads((tmp_path / "r1_A.json").read_text())
    b_path = tmp_path / "r1_B.json"; b = json.loads(b_path.read_text())
    b["response"] = a["response"]; b["response_sha"] = a["response_sha"]
    _write(b_path, b)
    assert validate_panel_bundle(r3, evidence_sha)[0] is False
