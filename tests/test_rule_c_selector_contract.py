import json
from pathlib import Path
import numpy as np
import pytest
from scripts.run_rule_c_selector import ratio_dials, exact_epochs, consensus, rule_c_select
from scripts.run_rule_c_offline import main as offline_main

def test_ratio_grid_and_consensus():
    assert ratio_dials(1000, [.005,.01,.02]) == [
        {"ratio":.005,"mcs":5,"ms":1,"method":"leaf","eps":.06},
        {"ratio":.01,"mcs":10,"ms":2,"method":"leaf","eps":.06},
        {"ratio":.02,"mcs":20,"ms":5,"method":"leaf","eps":.06}]
    assert consensus([{"selected_ep":4},{"selected_ep":4},{"selected_ep":5}]) == 4
    assert consensus([{"selected_ep":4},{"selected_ep":5},{"selected_ep":6}]) is None

def test_rule_c_k_q75_retention_and_deterministic_tie_break():
    rows={f"ep{i:02d}":{"k":10,"pre_reassign_noise":9,"over_merge":1,"stability":0,"coherence":0} for i in range(1,21)}
    for epoch,k,noise in ((1,1,0),(2,10,2),(3,10,1),(4,10,1)):
        rows[f"ep{epoch:02d}"]={"k":k,"pre_reassign_noise":noise,"over_merge":0,"stability":.75,"coherence":.80}
    result=rule_c_select(rows)
    assert result == {"gate_pass_epochs":[1,2,3,4],"k_q75":10.0,"retained_epochs":[2,3,4],"selected_ep":3,"tie_break":"lowest_epoch"}

def test_exact_ep_coverage_and_labeled_trap(tmp_path):
    d=tmp_path/"ck"; d.mkdir()
    for i in range(1,20): (d/f"proj_ep{i}.pt").write_text("x")
    assert not exact_epochs(d)
    labeled=tmp_path/"labeled.json"; labeled.write_text(json.dumps({"root":"x","files":[{"path":"a","label":"b"}]}))
    from scripts.run_rule_c_selector import main
    with pytest.raises(SystemExit, match="label-bearing"):
        main(["--pool",str(labeled),"--backbone",str(tmp_path/"bb.pt"),"--proj-dir",str(d),"--out-dir",str(tmp_path/"out")])

def test_offline_refuses_missing_or_mismatched_snapshot(tmp_path):
    labeled=tmp_path/"labeled.json"; labeled.write_text(json.dumps({"files":[{"path":"a","label":"b"}]}))
    with pytest.raises(SystemExit):
        offline_main(["--selection-snapshot",str(tmp_path/"none"),"--expected-selection-sha256","x","--labeled-pool",str(labeled),"--out",str(tmp_path/"o")])

def test_no_consensus_persists_diagnostic_snapshot_and_bundle(tmp_path, monkeypatch, capsys):
    from scripts import run_rule_c_selector as selector
    pool=tmp_path/"pool.json"; pool.write_text(json.dumps({"root":"images","files":["a.png"]}))
    proj=tmp_path/"proj"; proj.mkdir()
    for epoch in range(1,21): (proj/f"proj_ep{epoch}.pt").write_bytes(b"checkpoint")
    emb={"frozen":np.array([[0.0]]), **{f"ep{epoch:02d}":np.array([[float(epoch)]]) for epoch in range(1,21)}, **{f"z0_s{seed}":np.array([[float(seed)]]) for seed in range(1,11)}}
    monkeypatch.setattr(selector, "embeddings", lambda *args: emb)
    monkeypatch.setattr(selector, "label_free_metrics", lambda z,dial: ({"k":1,"pre_reassign_noise":0,"over_merge":0,"stability":.8,"coherence":.8}, np.array([0])))
    monkeypatch.setattr(selector, "rule_c_select", lambda rows: {"gate_pass_epochs":[],"k_q75":None,"retained_epochs":[],"selected_ep":None,"tie_break":"lowest_epoch"})
    out=tmp_path/"out"
    with pytest.raises(SystemExit, match="no 2/3 label-free epoch consensus"):
        selector.main(["--pool",str(pool),"--backbone",str(tmp_path/"backbone.pt"),"--proj-dir",str(proj),"--out-dir",str(out)])
    snap_path=out/"selection_snapshot.json"; bundle_path=out/"prediction_embedding_bundle.json"
    snap=json.loads(snap_path.read_text())
    assert snap["schema_version"] == "rule_c_label_free_selection.v2"
    assert snap["status"] == "diagnostic_no_consensus"
    assert snap["selection_valid"] is False
    assert snap["reason"] == "no 2/3 label-free epoch consensus"
    assert snap["consensus_selected_ep"] is None
    assert snap["per_dial"] and snap["sensitivity_audit"]["metrics"]
    assert bundle_path.is_file() and (out/"prediction_embedding_bundle.npz").is_file()
    assert snap["bundle_path"] == str(bundle_path.resolve())
    assert snap["bundle_sha256"] == selector.sha(bundle_path)
    printed=json.loads(capsys.readouterr().out)
    assert printed["selection_snapshot_path"] == str(snap_path.resolve())
    assert printed["selection_snapshot_sha256"] == selector.sha(snap_path)
    assert printed["status"] == "diagnostic_no_consensus"
