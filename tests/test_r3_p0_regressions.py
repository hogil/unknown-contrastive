import hashlib, json
from pathlib import Path
import numpy as np
import pytest

from scripts.run_unknown_campaign import Campaign, Registry, sha256_tree, run_rule_c_evaluations
from scripts.run_rule_c_offline import main as offline_main

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def test_direct_selected_dispatch_refuses_stop(monkeypatch, tmp_path):
    stop = tmp_path / "STOP_REQUESTED"; stop.write_text("keep")
    monkeypatch.setattr("scripts.run_unknown_campaign.STOP_FLAG", stop)
    with pytest.raises(ValueError, match="STOP_REQUESTED"):
        Campaign(registry_path=tmp_path / "registry.jsonl").run_selected(3, "strict_novel_base_seed42", 42, "x")
    assert stop.exists()

def test_base_lr_effective_configs_only_differ_in_allowed_fields():
    cfg=json.loads((ROOT / "configs" / "unknown_campaign_v1.json").read_text(encoding="utf-8"))
    steps={s["name"]:s for t in cfg["tiers"] for s in t.get("steps",[])}
    base, lr=steps["strict_novel_base_seed42"], steps["strict_novel_lr008_seed42"]
    def normalized(step):
        out={k:v for k,v in step.items() if k not in {"name","id"}}
        out.pop("decision_required_action", None)  # LR008 dispatch is adoption-gated.
        out["recipe"]={k:v for k,v in out["recipe"].items() if k != "lr_head"}
        out["env"]={k:v for k,v in out["env"].items() if k not in {"REPRO_LR","REPRO_TAG"}}
        out["rule_c"]={k:v for k,v in out["rule_c"].items() if k != "v3_r3_binding"}
        return out
    assert normalized(base) == normalized(lr)

def _offline_fixture(tmp_path, *, root="R", paths=("a","b","c","d")):
    labels=["A","A","B","B"]; manifest=tmp_path/"labeled.json"
    manifest.write_text(json.dumps({"root":root,"files":[{"path":p,"label":l,"block_id":f"block_{i}"} for i,(p,l) in enumerate(zip(paths,labels))]}))
    unlabeled=tmp_path/"unlabeled.json"; unlabeled.write_text(json.dumps({"root":root,"files":[{"path":p} for p in paths]}))
    arrays={"paths":np.asarray(paths)}
    pred=np.asarray([0,0,1,-1]); emb=np.asarray([[1.,0.],[.9,.1],[0.,1.],[.1,.9]],dtype="float32")
    for mcs,ms in ((2,1),(3,1),(4,1)):
        for name in ["frozen","ep02",*[f"z0_s{i}" for i in range(1,11)]]:
            arrays[f"{mcs}_{ms}_{name}"]=pred; arrays[name]=emb
    npz=tmp_path/"bundle.npz"; np.savez(npz,**arrays)
    bundle=tmp_path/"bundle.json"; bundle.write_text(json.dumps({"root":root,"paths":list(paths),"npz_path":str(npz),"npz_sha256":digest(npz)}))
    snap=tmp_path/"snapshot.json"; snap.write_text(json.dumps({"offline_labels_evaluated":False,"pool":str(unlabeled.resolve()),"pool_sha256":digest(unlabeled),"bundle_path":str(bundle),"bundle_sha256":digest(bundle),"consensus_selected_ep":2,"primary_dials":[{"ratio":.005},{"ratio":.01},{"ratio":.02}],"per_dial":[{"dial":{"ratio":.005,"mcs":2,"ms":1}},{"dial":{"ratio":.01,"mcs":3,"ms":1}},{"dial":{"ratio":.02,"mcs":4,"ms":1}}]}))
    return manifest,snap

def test_offline_refuses_legacy_v2_snapshot_before_opening_labels(tmp_path):
    labeled,snap=_offline_fixture(tmp_path); out=tmp_path/"offline.json"
    with pytest.raises(SystemExit):
        offline_main(["--selection-snapshot",str(snap),"--expected-selection-sha256",digest(snap),"--labeled-pool",str(labeled),"--out",str(out)])
    assert not out.exists()

@pytest.mark.parametrize("root,paths", [("OTHER",("a","b","c","d")), ("R",("b","a","c","d"))])
def test_offline_rejects_root_or_path_order_mismatch(tmp_path, root, paths):
    labeled,snap=_offline_fixture(tmp_path); altered=tmp_path/"bad.json"
    original=json.loads(labeled.read_text()); original["root"]=root; original["files"]=[{"path":p,"label":"A" if i<2 else "B"} for i,p in enumerate(paths)]; altered.write_text(json.dumps(original))
    with pytest.raises(SystemExit):
        offline_main(["--selection-snapshot",str(snap),"--expected-selection-sha256",digest(snap),"--labeled-pool",str(altered),"--out",str(tmp_path/"o")])

@pytest.mark.parametrize("evidence,config,valid", [("old","same",True),("same","other",True),("same","same",False)])
def test_lr_predecessor_requires_same_evidence_config_and_valid_artifact(monkeypatch, tmp_path, evidence, config, valid):
    registry=tmp_path/"registry.jsonl"; campaign=Campaign(registry_path=registry)
    monkeypatch.setattr("scripts.run_unknown_campaign.STOP_FLAG", tmp_path/"no_stop")
    panel=tmp_path/"r3.json"; panel.write_text("{}")
    queue=[{"id":"strict_novel_base_seed42","tier":3,"step":"strict_novel_base_seed42","seed":42},{"id":"strict_novel_lr008_seed42","tier":3,"step":"strict_novel_lr008_seed42","seed":42}]
    monkeypatch.setattr("scripts.run_unknown_campaign.read_json",lambda _: {"proposed_queue":queue})
    artifact=tmp_path/"artifact"; artifact.write_bytes(b"ok")
    Registry(registry).append({"tier":3,"step":"strict_novel_base_seed42","seed":42,"status":"completed","evidence_packet_sha":evidence,"config_snapshot_sha256":campaign.config_snapshot_sha256 if config=="same" else "other","artifact_path":str(artifact),"artifact_sha256":sha256_tree(artifact) if valid else "bad"})
    monkeypatch.setattr(campaign,"_acquire_lock",lambda:None); monkeypatch.setattr(campaign,"_release_lock",lambda:None); monkeypatch.setattr(campaign,"_reconcile_started_orphan",lambda *a:None); monkeypatch.setattr(campaign,"matching_final_record",lambda *a:None); monkeypatch.setattr(campaign,"_run_one_step",lambda *a,**k:{"status":"blocked"})
    if evidence=="same" and config=="same" and valid:
        assert campaign.run_selected(3,"strict_novel_lr008_seed42",42,"x",str(panel),"same")["status"] == "blocked"
    else:
        with pytest.raises(ValueError,match="queue order"):
            campaign.run_selected(3,"strict_novel_lr008_seed42",42,"x",str(panel),"same")

def test_rule_c_selector_then_offline_argv(monkeypatch, tmp_path):
    ck=tmp_path/"trainer"/"checkpoints"; ck.mkdir(parents=True)
    for i in range(1,21): (ck/f"proj_ep{i}.pt").write_text(str(i))
    seen=[]
    monkeypatch.setattr("scripts.run_unknown_campaign.STOP_FLAG", tmp_path/"no_stop")
    monkeypatch.setattr("scripts.run_unknown_campaign.check_gpu_guard", lambda _: (True, {}))
    def fake(argv, run_dir, *_a, **_k):
        seen.append((argv,_k.get("env",{}))); out=Path(argv[argv.index("--out-dir")+1]) if "--out-dir" in argv else None
        if out:
            out.mkdir(parents=True,exist_ok=True)
            npz=out/"bundle.npz"; npz.write_bytes(b"npz")
            bundle=out/"bundle.json"; bundle.write_text(json.dumps({"npz_path":str(npz),"npz_sha256":digest(npz)}))
            (out/"selection_snapshot.json").write_text(json.dumps({"schema_version":"rule_c_label_free_selection.v2","offline_labels_evaluated":False,"bundle_path":str(bundle),"bundle_sha256":digest(bundle)}))
        elif "run_rule_c_v3_reselector.py" in argv[1]:
            v3=Path(argv[argv.index("--out")+1]); seal=Path(argv[argv.index("--seal-out")+1])
            v3.write_text(json.dumps({"schema_version":"rule_c_label_free_selection.v3","status":"selected","selection_valid":True,"labels_used":False}))
            seal.write_text(json.dumps({"schema_version":"rule_c_epoch_seal.v3"}))
        else: Path(argv[argv.index("--out")+1]).write_text("{}")
        return {"status":"completed","returncode":0}
    monkeypatch.setattr("scripts.run_unknown_campaign.run_subprocess_step",fake)
    binding=tmp_path/"r3.json"; binding.write_text("{}")
    step={"name":"x","rule_c":{"backbone":"bb.pt","unlabeled_pool":"u.json","offline_pool":"l.json","v3_r3_binding":str(binding),"ratios":[.005,.01,.02],"z0_seeds":list(range(1,11)),"device":"cuda","batch_size":16}}
    arts,err=run_rule_c_evaluations(step,tmp_path/"trainer",tmp_path/"attempt","python",{"REPRO_GPU_MEMORY_FRACTION":"0.40"},1,{"gpu_memory_fraction":.40})
    first, v3, second = seen
    assert err is None and len(seen)==3 and "run_rule_c_selector.py" in first[0][1] and "run_rule_c_v3_reselector.py" in v3[0][1] and "run_rule_c_offline.py" in second[0][1]
    assert first[0][first[0].index("--device")+1]=="cuda" and first[0][first[0].index("--batch-size")+1]=="16" and first[0][first[0].index("--backbone")+1]=="bb.pt"
    assert second[0][second[0].index("--expected-selection-sha256")+1] == digest(tmp_path/"attempt"/"rule_c"/"selection_snapshot_v3.json") and second[0][second[0].index("--epoch-seal")+1].endswith("epoch_seal_v3.json") and v3[1]["CUDA_VISIBLE_DEVICES"] == second[1]["CUDA_VISIBLE_DEVICES"] == ""
