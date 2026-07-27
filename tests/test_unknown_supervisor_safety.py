from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_unknown_campaign import (
    Campaign,
    ExclusiveFileLock,
    Registry,
    build_three_seed_aggregate,
    preflight,
    run_rule_c_evaluations,
    sha256_file,
    sha256_tree,
    check_gpu_guard,
)
from scripts.run_unknown_supervisor import (
    create_evidence_packet,
    initial_state,
    materialize_initial_queue,
    materialize_post_gate_queue,
    tick,
    validate_dispatch_result,
    _step_for_item,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "unknown_campaign_v1.json"


def test_exact_pre_registered_cells_and_no_local_off():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    outer = cfg["outer_loop"]
    assert [x["id"] for x in outer["initial_queue"]["cells"]] == [
        "strict_novel_base_seed42", "strict_novel_lr008_seed42"]
    assert [x["id"] for x in outer["post_gate_cells"]] == [
        "strict_novel_t010_seed42", "strict_novel_t030_seed42",
        "strict_novel_neg060_seed42", "strict_novel_neg085_seed42",
        "strict_novel_q4096_seed42", "strict_novel_q32768_seed42",
        "strict_novel_lr002_seed42",
    ]
    assert "local_off" not in json.dumps(outer).casefold()
    steps = {step["name"]: step for tier in cfg["tiers"] for step in tier.get("steps", [])}
    for cell in outer["post_gate_cells"]:
        step = steps[cell["step"]]
        assert step["decision_required_action"] == "next_queue"
        assert step["recipe"]["epochs"] == 20
        assert step["env"]["REPRO_EPOCHS"] == "20"

    tier3_cells = outer["initial_queue"]["cells"] + outer["post_gate_cells"]
    assert len(tier3_cells) == 9
    for cell in tier3_cells:
        step = steps[cell["step"]]
        assert step["env"]["REPRO_WORKERS"] == "8"
        assert step["command"] == ["{python}", "_may_ablation.py", "B4"]
        assert step["env"]["REPRO_GPU_MEMORY_FRACTION"] == "0.40"


def test_evidence_uses_single_recompute_data_pass_for_manifest_content(monkeypatch, tmp_path):
    cfg = {"campaign_id": "test", "tiers": [{"tier": 1, "steps": [{
        "name": "cell", "recipe": {}, "env": {}, "command": []}]}],
        "outer_loop": {"initial_queue": {"cells": [{"id": "cell", "tier": 1, "step": "cell", "seed": 1}]}}}
    config = tmp_path / "config.json"; config.write_text(json.dumps(cfg))
    state = {"outer_loop": cfg["outer_loop"], "completed": [], "failed": [], "dispatch_failures": []}
    data = [{"manifest": str(tmp_path / "pool.json"), "manifest_sha256": "manifest-sha",
             "root": str(tmp_path / "root"), "count": 3, "content_sha256": "content-sha"}]
    binding = {"recompute_contract": {"data": data}, "source_snapshot_sha256": "source",
               "data_snapshot_sha256": "data", "backbone_snapshot_sha256": "backbone",
               "env_snapshot_sha256": "env", "config_snapshot_sha256": "config",
               "ordered_queue_sha256": "queue"}
    calls = []
    monkeypatch.setattr("scripts.run_unknown_supervisor.ensure_split_overlap_audits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.recompute_contract", lambda *_args: calls.append(1) or binding)
    monkeypatch.setattr("scripts.run_unknown_supervisor._git_snapshot", lambda: {})
    monkeypatch.setattr("scripts.run_unknown_supervisor.PANEL_ROOT", tmp_path / "panels")
    monkeypatch.setattr("scripts.run_unknown_supervisor.subprocess.run", lambda *_args, **_kwargs: type("P", (), {"stdout": ""})())
    panel = create_evidence_packet(state, cfg, config, "experiment_design")
    evidence = json.loads(Path(panel["evidence_packet_path"]).read_text())
    assert calls == [1]
    assert evidence["manifest_content"] == [{"manifest": data[0]["manifest"], "manifest_sha256": "manifest-sha",
                                               "root": data[0]["root"], "image_count": 3,
                                               "image_content_sha256": "content-sha"}]


@pytest.mark.parametrize("source", ["manifest", "pool", "command", "env"])
@pytest.mark.parametrize("variant", ["relative", "dot", "absolute", "case"])
def test_sealed_variants_fail_closed(tmp_path, source, variant):
    sealed_rel = "data/pools/v2/unknown/strict_novel_test.json"
    value = {
        "relative": sealed_rel,
        "dot": "./data/pools/v2/unknown/../unknown/strict_novel_test.json",
        "absolute": str((REPO / sealed_rel).resolve()),
        "case": str((REPO / sealed_rel).resolve()).upper(),
    }[variant]
    cfg = {
        "sealed_test_pools": {"unknown": sealed_rel},
        "safety": {"min_free_gb": {"D": 0}},
    }
    step = {}
    if source in {"manifest", "pool"}:
        step[source] = value
    elif source == "command":
        step["command"] = ["python", "--pool", value]
    else:
        step["env"] = {"REPRO_DATA": value}
    ok, blockers, _ = preflight(step, cfg)
    assert not ok and blockers == ["sealed_test_reference"]


def test_manifest_root_sealed_fails_closed(tmp_path):
    sealed = tmp_path / "sealed"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"root": str(sealed)}), encoding="utf-8")
    cfg = {"sealed_test_pools": {"unknown": str(sealed)},
           "safety": {"min_free_gb": {"D": 0}, "allowlist_roots": [str(tmp_path)]}}
    ok, blockers, detail = preflight({"manifest": str(manifest)}, cfg)
    assert not ok and blockers == ["sealed_test_reference"]
    assert detail["source"] == "manifest.root"


def test_gpu_fraction_child_hook_is_mandatory(monkeypatch):
    monkeypatch.setattr("scripts.run_unknown_campaign.check_disk_guard", lambda _: (True, {}))
    monkeypatch.setattr("scripts.run_unknown_campaign.check_gpu_guard", lambda _: (True, {}))
    cfg = {"safety": {"min_free_gb": {}, "gpu_memory_fraction": 0.40}}
    ok, blockers, _ = preflight({"needs_gpu": True, "env": {}}, cfg)
    assert not ok and "gpu_memory_fraction_unsupported" in blockers
    ok, blockers, _ = preflight(
        {"needs_gpu": True, "env": {"REPRO_GPU_MEMORY_FRACTION": "0.40"}}, cfg)
    assert not ok and "gpu_memory_hook_missing_or_late" in blockers
    ok, blockers, _ = preflight({
        "needs_gpu": True,
        "command": ["{python}", "_may_ablation.py", "B4"],
        "env": {"REPRO_GPU_MEMORY_FRACTION": "0.40"},
    }, cfg)
    assert ok and blockers == []


def test_rule_c_hookless_selector_fails_closed(monkeypatch, tmp_path):
    checkpoints = tmp_path / "trainer" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in range(1, 21):
        (checkpoints / f"proj_ep{epoch}.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "scripts.run_unknown_campaign.rule_c_selector_gpu_memory_hook_contract_ok", lambda: False)
    monkeypatch.setattr("scripts.run_unknown_campaign.STOP_FLAG", tmp_path / "no_stop")
    artifacts, error = run_rule_c_evaluations(
        {"rule_c": {"device": "cuda"}}, tmp_path / "trainer", tmp_path / "attempt", "python",
        {"REPRO_GPU_MEMORY_FRACTION": "0.40"}, 1, {"gpu_memory_fraction": 0.40})
    assert artifacts == []
    assert error == "rule_c_selector_gpu_memory_hook_missing_or_late"


def test_gpu_pid_probe_failure_fails_closed(monkeypatch):
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: None)
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_total_mb", lambda: None)
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("missing")))
    ok, detail = check_gpu_guard({"gpu_max_concurrent": 1})
    assert not ok and detail["gpu_pid_probe_ok"] is False


@pytest.mark.parametrize("name,expected", [("explorer.exe", True), ("python.exe", False), ("llama-server.exe", False), ("mystery.exe", False)])
def test_wddm_gpu_pid_names_fail_closed_except_explicit_gui_allowlist(monkeypatch, name, expected):
    from types import SimpleNamespace
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: 12500)
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_total_mb", lambda: 16380)
    monkeypatch.setattr("scripts.run_unknown_campaign.scan_running_locks", lambda _: [])
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="123, N/A\n"))
    class P:
        def __init__(self, pid): pass
        def name(self): return name
    monkeypatch.setattr("psutil.Process", P)
    ok, _ = check_gpu_guard({
        "gpu_max_concurrent": 1,
        "gpu_memory_fraction": 0.40,
        "gpu_headroom_mb": 1024,
        "gpu_non_compute_process_allowlist": ["explorer.exe"],
    })
    assert ok is expected


def test_wddm_allowlist_normalizes_exact_basenames(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: 12500)
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_total_mb", lambda: 16380)
    monkeypatch.setattr("scripts.run_unknown_campaign.scan_running_locks", lambda _: [])
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="7, N/A\n"))
    class P:
        def __init__(self, pid): pass
        def name(self): return "ShellExperienceHost.EXE"
    monkeypatch.setattr("psutil.Process", P)
    ok, _ = check_gpu_guard({
        "gpu_memory_fraction": 0.40,
        "gpu_headroom_mb": 1024,
        "gpu_non_compute_process_allowlist": ["shellexperiencehost.exe"],
    })
    assert ok


def test_allowed_external_compute_coexists_when_capacity_is_available(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: 12500)
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_total_mb", lambda: 16380)
    monkeypatch.setattr("scripts.run_unknown_campaign.scan_running_locks", lambda _: [])
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="11, N/A\n12, N/A\n"))
    class P:
        def __init__(self, pid): self.pid = pid
        def name(self): return "ollama.exe" if self.pid == 11 else "llama-server.exe"
    monkeypatch.setattr("psutil.Process", P)
    safety = {"gpu_memory_fraction": 0.40, "gpu_headroom_mb": 1024,
              "gpu_coexist_process_allowlist": ["llama-server.exe", "ollama.exe"]}
    ok, detail = check_gpu_guard(safety)
    assert ok
    assert detail["required_free_mb"] == pytest.approx(7576)
    assert [p["classification"] for p in detail["gpu_processes"]] == ["coexist_allowlisted"] * 2


def test_allowed_external_compute_blocks_when_capacity_or_project_lock_is_unavailable(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_total_mb", lambda: 16380)
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="11, N/A\n"))
    class P:
        def __init__(self, _pid): pass
        def name(self): return "ollama.exe"
    monkeypatch.setattr("psutil.Process", P)
    safety = {"gpu_memory_fraction": 0.40, "gpu_headroom_mb": 1024,
              "gpu_coexist_process_allowlist": ["ollama.exe"]}
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: 7575)
    monkeypatch.setattr("scripts.run_unknown_campaign.scan_running_locks", lambda _: [])
    ok, detail = check_gpu_guard(safety)
    assert not ok and detail["busy_by_mem"] is True
    monkeypatch.setattr("scripts.run_unknown_campaign.gpu_free_mb", lambda: 12500)
    monkeypatch.setattr("scripts.run_unknown_campaign.scan_running_locks", lambda _: [{"alive": True, "pid": 99}])
    ok, detail = check_gpu_guard(safety)
    assert not ok and detail["busy_by_lock"] is True


def test_v2_running_batch_without_snapshot_is_quarantined(monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    state.update({"state": "RUNNING_BATCH", "config_snapshot_sha256": None,
                  "queue": [{"id": "x", "tier": 3, "step": "strict_novel_base_seed42", "seed": 42}]})
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state", lambda _: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.event", lambda *a, **k: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG", tmp_path / "no_stop")
    assert tick(state, cfg, CONFIG) is False
    assert state["state"] == "REPAIR_EVIDENCE" and state["queue"] == []


def test_exclusive_lock_race_and_owner_verified_release(tmp_path):
    path = tmp_path / "lock"
    first, second = ExclusiveFileLock(path), ExclusiveFileLock(path)
    first.acquire()
    with pytest.raises(RuntimeError):
        second.acquire()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["owner_token"] = "other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert first.release() is False
    assert path.exists()


def test_resume_requires_matching_fingerprint_and_artifact(tmp_path):
    registry = tmp_path / "registry.jsonl"
    campaign = Campaign(registry_path=registry)
    _, step = campaign._find_step(3, "strict_novel_base_seed42")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable")
    Registry(registry).append({
        "campaign_id": campaign.campaign_id,
        "config_snapshot_sha256": campaign.config_snapshot_sha256,
        "step_fingerprint": campaign._step_fingerprint(3, step, 42),
        "tier": 3, "step": step["name"], "seed": 42, "status": "completed",
        "artifact_path": str(artifact), "artifact_sha256": sha256_file(artifact),
    })
    assert campaign.matching_final_record(3, step, 42) is not None
    changed = {**step, "recipe": {**step["recipe"], "epochs": 5}}
    assert campaign.matching_final_record(3, changed, 42) is None
    artifact.write_bytes(b"changed")
    assert campaign.matching_final_record(3, step, 42) is None


def test_three_seed_aggregate_requires_paired_deltas_and_records_mean_std():
    records = [
        {"seed": seed, "run_id": f"s{seed}",
         "metrics": {"ari": {"drop": value}, "paired_deltas": {"ari": value - 0.1}}}
        for seed, value in zip((1, 2, 42), (0.1, 0.2, 0.3))
    ]
    aggregate, error = build_three_seed_aggregate(records)
    assert error is None
    assert aggregate["metric_statistics"]["ari.drop"]["mean"] == pytest.approx(0.2)
    assert aggregate["metric_statistics"]["ari.drop"]["std"] == pytest.approx(0.1)
    assert aggregate["paired_deltas"]["ari"]["values"] == pytest.approx([0.0, 0.1, 0.2])


def test_rule_c_requires_exact_ep1_to_ep20_coverage(tmp_path, monkeypatch):
    trainer = tmp_path / "trainer"
    checkpoints = trainer / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in (1, 2, 3):
        (checkpoints / f"proj_ep{epoch}.pt").write_bytes(str(epoch).encode())
    seen = []

    def fake_run(argv, run_dir, *_args, **_kwargs):
        seen.append(argv)
        out_dir = Path(argv[argv.index("--out-dir") + 1])
        out_name = argv[argv.index("--out-name") + 1]
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot = out_dir / f"{out_name}.selection.json"
        snapshot.write_text("{}", encoding="utf-8")
        result = {
            "select_rule": "rich_noise",
            "all_eps": {f"ep{x:02d}": {} for x in (1, 2, 3)},
            "selection_snapshot_sha256": sha256_file(snapshot),
        }
        (out_dir / f"{out_name}.json").write_text(json.dumps(result), encoding="utf-8")
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr("scripts.run_unknown_campaign.run_subprocess_step", fake_run)
    step = {"name": "cell", "rule_c": {
        "backbone": "bb.pt", "pool": "val.json",
        "dials": [{"mcs": 6, "ms": 3}, {"mcs": 42, "ms": 10}]}}
    artifacts, error = run_rule_c_evaluations(
        step, trainer, tmp_path / "attempt", "python", {}, 1)
    assert artifacts == []
    assert error == "rule_c_requires_exact_proj_ep1_to_ep20"


def test_post_gate_materialization_requires_panel_and_keeps_concrete_items():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    materialize_initial_queue(state)
    assert state["queue"] == []
    state["panel"] = {"approved": True, "action": "experiment_design",
                      "r3_artifact_path": "initial_r3_C.json", "evidence_packet_sha": "initial-sha"}
    materialize_initial_queue(state)
    assert len(state["queue"]) == 2
    state["initial_gate_passed"] = True
    state["initial_design_approval_sha"] = "initial-sha"
    state["completed"] = [{"id": "strict_novel_base_seed42"}, {"id": "strict_novel_lr008_seed42"}]
    materialize_post_gate_queue(state)
    assert len(state["queue"]) == 2
    state["panel"] = {"approved": True, "action": "next_queue",
                      "r3_artifact_path": "r3_C.json", "evidence_packet_sha": "next-sha"}
    materialize_post_gate_queue(state)
    assert len(state["queue"]) == 9
    assert all({"tier", "step", "seed"}.issubset(item) for item in state["queue"])


@pytest.mark.parametrize("passed", [True, False])
def test_scoring_records_actual_initial_screen_outcome(monkeypatch, tmp_path, passed):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    state["state"] = "SCORING"
    state["completed"] = [{"id": cell["id"]} for cell in cfg["outer_loop"]["initial_queue"]["cells"]]
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG", tmp_path / "no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.resource_blocked", lambda _: (False, {}))
    monkeypatch.setattr("scripts.run_unknown_supervisor.evaluate_initial_screen", lambda _: {"passed": passed})
    monkeypatch.setattr("scripts.run_unknown_supervisor.create_evidence_packet", lambda *args: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.event", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state", lambda _: None)
    assert tick(state, cfg, CONFIG) is False
    assert state["state"] == "WAITING_PANEL"
    assert state["initial_gate_passed"] is passed


def test_next_queue_panel_cannot_flip_failed_initial_gate(monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    state.update({"state": "WAITING_PANEL", "initial_gate_passed": False,
                  "completed": [{"id": cell["id"]} for cell in cfg["outer_loop"]["initial_queue"]["cells"]],
                  "panel": {"action": "next_queue", "approved": True}})
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG", tmp_path / "no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.resource_blocked", lambda _: (False, {}))
    monkeypatch.setattr("scripts.run_unknown_supervisor.ingest_panel", lambda _: (True, "approved"))
    monkeypatch.setattr("scripts.run_unknown_supervisor.event", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state", lambda _: None)
    tick(state, cfg, CONFIG)
    assert state["state"] == "WAITING_PANEL" and state["initial_gate_passed"] is False and state["queue"] == []


def test_failed_initial_screen_creates_empty_screen_review_evidence(monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    state["state"] = "SCORING"
    state["completed"] = [{"id": cell["id"]} for cell in cfg["outer_loop"]["initial_queue"]["cells"]]
    captured = {}
    def fake_evidence(_state, _cfg, _config, action):
        captured.update({"action": action, "proposed_queue": [] if action == "screen_review" else ["unexpected"]})
        return {}
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG", tmp_path / "no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.resource_blocked", lambda _: (False, {}))
    monkeypatch.setattr("scripts.run_unknown_supervisor.evaluate_initial_screen", lambda _: {"passed": False})
    monkeypatch.setattr("scripts.run_unknown_supervisor.create_evidence_packet", fake_evidence)
    monkeypatch.setattr("scripts.run_unknown_supervisor.event", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state", lambda _: None)
    assert tick(state, cfg, CONFIG) is False
    assert captured == {"action": "screen_review", "proposed_queue": []}


def test_running_batch_resource_wait_preserves_queue_and_attempt(monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = initial_state(cfg)
    queued = {"id": "strict_novel_base_seed42", "tier": 3, "step": "strict_novel_base_seed42", "seed": 42,
              "attempts": 0, "panel_r3_artifact": "R3_C.json", "evidence_packet_sha": "evidence"}
    state.update({"state": "RUNNING_BATCH", "attempt": 7, "queue": [dict(queued)],
                  "config_snapshot_sha256": sha256_file(CONFIG)})
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG", tmp_path / "no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.resource_blocked", lambda _: (True, {"gpu": {"ok": False}}))
    monkeypatch.setattr("scripts.run_unknown_supervisor.subprocess.run", lambda *args, **kwargs: pytest.fail("executor must not run while resources are blocked"))
    monkeypatch.setattr("scripts.run_unknown_supervisor.event", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state", lambda _: None)
    assert tick(state, cfg, CONFIG) is False
    assert state["state"] == "RUNNING_BATCH" and state["attempt"] == 7 and state["queue"] == [queued]


def test_supervisor_step_fingerprint_matches_run_selected_effective_step(tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    panel = tmp_path / "R3_C.json"; panel.write_text("{}")
    item = {"id": "strict_novel_base_seed42", "tier": 3, "step": "strict_novel_base_seed42", "seed": 42,
            "panel_r3_artifact": str(panel), "evidence_packet_sha": "evidence", "approved_queue_index": 0}
    campaign = Campaign(config_path=CONFIG, registry_path=tmp_path / "registry.jsonl")
    _, configured = campaign._find_step(item["tier"], item["step"])
    run_selected_effective = {**configured, "panel_r3_artifact": str(panel), "evidence_packet_sha": "evidence",
                              "dispatch_item": {key: item[key] for key in ("id", "tier", "step", "seed")},
                              "approved_queue_index": 0}
    assert campaign._step_fingerprint(item["tier"], _step_for_item(cfg, item), item["seed"]) == campaign._step_fingerprint(item["tier"], run_selected_effective, item["seed"])


@pytest.mark.parametrize("mutate", ["schema", "config"])
def test_schema_or_config_drift_archives_and_resets_state(monkeypatch, tmp_path, mutate):
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); state=initial_state(cfg)
    state.update({"state":"WAITING_PANEL","queue":[{"id":"old"}],"schema_version":"legacy" if mutate=="schema" else "unknown_supervisor.v2","config_snapshot_sha256":"old" if mutate=="config" else None})
    root=tmp_path/"state"; monkeypatch.setattr("scripts.run_unknown_supervisor.STATE_ROOT",root)
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG",tmp_path/"no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.event",lambda *args,**kwargs:None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state",lambda _:None)
    assert tick(state,cfg,CONFIG) is False
    assert state["state"]=="REPAIR_EVIDENCE" and state["queue"]==[] and state["config_snapshot_sha256"]==sha256_file(CONFIG)
    archived=Path(state["quarantined_previous_state"]["path"]); assert archived.is_file() and json.loads(archived.read_text())["queue"]==[{"id":"old"}]


def test_exact_post_gate_completion_requests_expansion_review(monkeypatch, tmp_path):
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); state=initial_state(cfg); state["state"]="SCORING"
    state["completed"]=[{"id":cell["id"]} for cell in cfg["outer_loop"]["initial_queue"]["cells"]+cfg["outer_loop"]["post_gate_cells"]]
    seen=[]
    monkeypatch.setattr("scripts.run_unknown_supervisor.STOP_FLAG",tmp_path/"no_stop")
    monkeypatch.setattr("scripts.run_unknown_supervisor.resource_blocked",lambda _:(False,{}))
    monkeypatch.setattr("scripts.run_unknown_supervisor.create_evidence_packet",lambda _s,_c,_p,action:seen.append(action))
    monkeypatch.setattr("scripts.run_unknown_supervisor.event",lambda *args,**kwargs:None)
    monkeypatch.setattr("scripts.run_unknown_supervisor.save_state",lambda _:None)
    assert tick(state,cfg,CONFIG) is False and seen==["expansion_review"] and state["state"]=="WAITING_PANEL"


@pytest.mark.parametrize("field,value,reason", [
    ("evidence_packet_sha", "wrong", "evidence_packet_sha_mismatch"),
    ("panel_r3_artifact_sha256", "wrong", "panel_r3_artifact_sha_mismatch"),
    ("approved_queue_index", 9, "approved_queue_index_mismatch"),
])
def test_dispatch_result_requires_panel_metadata_binding(tmp_path, field, value, reason):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = tmp_path / "registry.jsonl"; panel = tmp_path / "R3_C.json"; panel.write_text("{}")
    artifact = tmp_path / "artifact"; artifact.mkdir(); (artifact / "result.json").write_text("{}")
    item = {"id": "strict_novel_base_seed42", "tier": 3, "step": "strict_novel_base_seed42", "seed": 42,
            "panel_r3_artifact": str(panel), "evidence_packet_sha": "evidence", "approved_queue_index": 0}
    campaign = Campaign(config_path=CONFIG, registry_path=registry)
    bound_step = _step_for_item(cfg, item)
    record = {"dispatch_id": "dispatch", "tier": 3, "step": item["step"], "seed": 42, "campaign_id": cfg["campaign_id"],
              "config_snapshot_sha256": sha256_file(CONFIG), "step_fingerprint": campaign._step_fingerprint(3, bound_step, 42),
              "evidence_packet_sha": "evidence", "panel_r3_artifact_sha256": sha256_file(panel), "approved_queue_index": 0,
              "status": "completed", "artifact_path": str(artifact), "artifact_sha256": sha256_tree(artifact)}
    record[field] = value
    Registry(registry).append(record)
    _, failure = validate_dispatch_result(cfg, CONFIG, item, "dispatch", 0, registry)
    assert failure == reason
