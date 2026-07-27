import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.run_unknown_campaign import sha256_file, sha256_obj
from scripts.run_unknown_campaign import validate_panel_dispatch
from scripts.run_unknown_supervisor import evaluate_initial_screen

def _completed(tmp_path, name):
    sel=tmp_path/f"{name}.selector.json"; off=tmp_path/f"{name}.offline.json"
    sel.write_text("{}",encoding="utf-8"); off.write_text("{}",encoding="utf-8")
    return {"id":name,"rule_c_artifacts":[{"selection_snapshot_path":str(sel),"selection_snapshot_sha256":sha256_file(sel),"offline_path":str(off),"offline_sha256":sha256_file(off)}]}

def test_initial_screen_uses_exact_two_hash_valid_artifacts(monkeypatch,tmp_path):
    base,lr=_completed(tmp_path,"base"),_completed(tmp_path,"lr")
    state={"completed":[base,lr],"outer_loop":{"initial_queue":{"cells":[{"id":"base"},{"id":"lr"}]},"initial_screen":{}}}
    screen_root=tmp_path/"screens"; monkeypatch.setattr("scripts.run_unknown_supervisor.STATE_ROOT",screen_root)
    def fake(cmd, cwd):
        out=Path(cmd[cmd.index("--out")+1]); expected=json.loads(Path(cmd[cmd.index("--expected-shas")+1]).read_text()); out.write_text(json.dumps({"promotes":False,"gates":{"a":True},"input_sha256":expected})); return SimpleNamespace(returncode=0)
    monkeypatch.setattr("scripts.run_unknown_supervisor.subprocess.run",fake)
    result=evaluate_initial_screen(state)
    assert result["passed"] and Path(result["artifact_path"]).is_file()

@pytest.mark.parametrize("mode",["missing","tampered","multiple"])
def test_initial_screen_rejects_invalid_rule_c_artifact(monkeypatch,tmp_path,mode):
    base,lr=_completed(tmp_path,"base"),_completed(tmp_path,"lr")
    if mode=="missing": base["rule_c_artifacts"]=[]
    elif mode=="multiple": base["rule_c_artifacts"]*=2
    else: base["rule_c_artifacts"][0]["offline_sha256"]="bad"
    state={"completed":[base,lr],"outer_loop":{"initial_queue":{"cells":[{"id":"base"},{"id":"lr"}]}}}
    with pytest.raises(ValueError): evaluate_initial_screen(state)

@pytest.mark.parametrize("screen,reason", [
    (None,"panel_initial_screen_not_passed"), ({"passed":False,"promotes":False},"panel_initial_screen_not_passed"),
])
def test_next_queue_panel_requires_passed_screen(monkeypatch,tmp_path,screen,reason):
    item={"id":"x","tier":3,"step":"x","seed":1}; binding={"config_snapshot_sha256":"c","ordered_queue_sha256":"q"}
    evidence={"action":"next_queue","proposed_queue":[item],"binding":binding}
    if screen is not None: evidence["initial_screen"]=screen
    monkeypatch.setattr("scripts.run_unknown_campaign.validate_panel_bundle",lambda *_:(True,"ok",{})); monkeypatch.setattr("scripts.run_unknown_campaign.read_json",lambda _:evidence); monkeypatch.setattr("scripts.run_unknown_campaign.sha256_obj",lambda _:"q")
    ok,got=validate_panel_dispatch(tmp_path/"r3",None,item,"next_queue","c",{"outer_loop":{"post_gate_cells":[item]}})
    assert not ok and got==reason

def test_next_queue_panel_rejects_tampered_or_accepts_valid_screen(monkeypatch,tmp_path):
    cfg=json.loads((Path(__file__).resolve().parents[1]/"configs"/"unknown_campaign_v1.json").read_text())
    cells=cfg["outer_loop"]["initial_queue"]["cells"]; queue=cfg["outer_loop"]["post_gate_cells"]; item=queue[0]
    inputs={}; completed=[]
    for index,cell in enumerate(cells):
        step=next(s for t in cfg["tiers"] for s in t["steps"] if s["name"]==cell["step"]); unlabeled=(Path(__file__).resolve().parents[1]/step["rule_c"]["unlabeled_pool"]).resolve(); labeled=(Path(__file__).resolve().parents[1]/step["rule_c"]["offline_pool"]).resolve()
        bundle=tmp_path/f"{cell['id']}.bundle.json"; npz=tmp_path/f"{cell['id']}.bundle.npz"; bundle.write_text("{}");npz.write_text("npz")
        selector=tmp_path/f"{cell['id']}.selector.json"; offline=tmp_path/f"{cell['id']}.offline.json"
        selector.write_text(json.dumps({"pool":str(unlabeled),"pool_sha256":sha256_file(unlabeled),"bundle_path":str(bundle),"bundle_sha256":sha256_file(bundle),"consensus_selected_ep":2}))
        offline.write_text(json.dumps({"selection_snapshot_sha256":sha256_file(selector),"unlabeled_pool_path":str(unlabeled),"unlabeled_pool_sha256":sha256_file(unlabeled),"labeled_pool_path":str(labeled),"labeled_pool_sha256":sha256_file(labeled),"bundle_path":str(bundle),"bundle_sha256":sha256_file(bundle),"npz_path":str(npz),"npz_sha256":sha256_file(npz),"selected_ep":2}))
        prefix=("base" if index == 0 else "lr")
        inputs[f"{prefix}_selector"] = sha256_file(selector)
        inputs[f"{prefix}_offline"] = sha256_file(offline)
        completed.append({"id":cell["id"],"rule_c_artifacts":[{"selection_snapshot_path":str(selector),"selection_snapshot_sha256":sha256_file(selector),"offline_path":str(offline),"offline_sha256":sha256_file(offline)}]})
    artifact=tmp_path/"screen.json"; sidecar=tmp_path/"screen.input_sha256.json"
    artifact.write_text(json.dumps({"promotes":False,"gates":{"all":True},"input_sha256":inputs}))
    sidecar.write_text(json.dumps(inputs))
    screen={"passed":True,"promotes":False,"gates":{"all":True},"artifact_path":str(artifact),"artifact_sha256":sha256_file(artifact),"input_sha256":inputs,"expected_input_shas_path":str(sidecar),"expected_input_shas_sha256":sha256_file(sidecar)}
    evidence={"action":"next_queue","proposed_queue":queue,"binding":{"config_snapshot_sha256":"c","ordered_queue_sha256":sha256_obj(queue),"source_snapshot_sha256":"x","data_snapshot_sha256":"x","backbone_snapshot_sha256":"x","env_snapshot_sha256":"x"},"initial_screen":screen,"completed":completed}
    panel_dir=tmp_path/"panel"; panel_dir.mkdir(); r3=panel_dir/"R3_C.json"; r3.write_text("{}")
    (panel_dir/"evidence_packet.json").write_text(json.dumps(evidence))
    monkeypatch.setattr("scripts.run_unknown_campaign.validate_panel_bundle",lambda *_:(True,"ok",{}))
    monkeypatch.setattr("scripts.run_unknown_campaign.recompute_contract",lambda *_:{"source_snapshot_sha256":"x","data_snapshot_sha256":"x","backbone_snapshot_sha256":"x","env_snapshot_sha256":"x","ordered_queue_sha256":sha256_obj(queue)})
    assert validate_panel_dispatch(r3,None,item,"next_queue","c",cfg)[0]

    payload=json.loads(artifact.read_text()); payload["input_sha256"]={**inputs,"tampered":"x"}; artifact.write_text(json.dumps(payload))
    evidence["initial_screen"]["artifact_sha256"]=sha256_file(artifact); (panel_dir/"evidence_packet.json").write_text(json.dumps(evidence))
    assert validate_panel_dispatch(r3,None,item,"next_queue","c",cfg)[1]=="panel_initial_screen_payload_mismatch"

    artifact.write_text(json.dumps({"promotes":False,"gates":{"all":True},"input_sha256":inputs})); evidence["initial_screen"]["artifact_sha256"]=sha256_file(artifact)
    sidecar.write_text(json.dumps({**inputs,"tampered":"x"})); evidence["initial_screen"]["expected_input_shas_sha256"]=sha256_file(sidecar); (panel_dir/"evidence_packet.json").write_text(json.dumps(evidence))
    assert validate_panel_dispatch(r3,None,item,"next_queue","c",cfg)[1]=="panel_initial_screen_sidecar_mismatch"

    sidecar.write_text(json.dumps(inputs)); evidence["initial_screen"]["expected_input_shas_sha256"]=sha256_file(sidecar); (panel_dir/"evidence_packet.json").write_text(json.dumps(evidence))
    selector=Path(completed[0]["rule_c_artifacts"][0]["selection_snapshot_path"]); selector.write_text("tampered")
    assert validate_panel_dispatch(r3,None,item,"next_queue","c",cfg)[1]=="panel_initial_screen_source_artifact_drift"
