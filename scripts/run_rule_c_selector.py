"""Label-free Rule-C selector. It never reads labels or parent folder names."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, uuid, sys
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def atomic(path, value):
    tmp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp"); tmp.write_text(json.dumps(value,sort_keys=True,indent=2),encoding="utf-8"); os.replace(tmp,path); return sha(path)
def ratio_dials(n, ratios):
    return [{"ratio":r,"mcs":max(2,math.ceil(n*r)),"ms":max(1,math.floor(max(2,math.ceil(n*r))/4)),"method":"leaf","eps":.06} for r in ratios]
def exact_epochs(proj_dir):
    got={int(m.group(1)) for p in Path(proj_dir).glob("proj_ep*.pt") if (m:=re.search(r"proj_ep(\d+)\.pt$",p.name))}
    return got == set(range(1,21))
def consensus(per_dial):
    choices=[x.get("selected_ep") for x in per_dial if x.get("selected_ep") is not None]
    return next((e for e in sorted(set(choices)) if choices.count(e)>=2),None)
def rule_c_select(rows):
    """Apply the label-free Rule-C epoch choice to one run/dial."""
    candidates=[]
    for e in range(1,21):
        metric=rows[f"ep{e:02d}"]
        if metric["over_merge"]==0 and metric["stability"]>=.75 and metric["coherence"]>=.80:
            candidates.append((e,float(metric["k"]),float(metric["pre_reassign_noise"])))
    if not candidates:
        return {"gate_pass_epochs":[],"k_q75":None,"retained_epochs":[],"selected_ep":None,
                "tie_break":"lowest_epoch"}
    k_q75=float(np.percentile([c[1] for c in candidates],75))
    retained=[c for c in candidates if c[1]>=k_q75]
    selected=min(retained,key=lambda c:(c[2],c[0]))[0]
    return {"gate_pass_epochs":[c[0] for c in candidates],"k_q75":k_q75,
            "retained_epochs":[c[0] for c in retained],"selected_ep":selected,
            "tie_break":"lowest_epoch"}
def label_free_metrics(z, dial):
    from _grouping_eval import DIAL, label_free
    DIAL.update(mcs=dial["mcs"], ms=dial["ms"], eps=dial["eps"], method=dial["method"])
    value=label_free(z); pred=value.pop("pred")
    value["pre_reassign_noise"]=value.pop("noise_pct")
    value["cluster_size_distribution"]=sorted(int((pred==c).sum()) for c in set(pred.tolist()) if c != -1)
    return value,pred
def load_paths(manifest):
    if "label" in manifest or any(isinstance(x,dict) and "label" in x for x in manifest.get("files",[])): raise ValueError("label-bearing manifest refused")
    paths=manifest.get("files")
    if not isinstance(manifest.get("root"),str) or not isinstance(paths,list) or not all(isinstance(x,str) for x in paths): raise ValueError("path-only manifest required")
    root=Path(manifest["root"]); return [str(root / p) for p in paths],paths
def embeddings(backbone, proj_dir, paths, cache, device, batch_size):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    import _grouping_eval as core
    core.DEV=device
    if cache and cache.is_file(): raw=torch.from_numpy(np.load(cache)).float()
    else:
        raw=core.extract_backbone_features(paths,core.load_backbone(backbone),batch_size=batch_size)
        if cache: cache.parent.mkdir(parents=True,exist_ok=True); np.save(cache,raw.numpy())
    out={"frozen":core.embedding_from_features(raw,None)}
    for e in range(1,21):
        d=torch.load(proj_dir/f"proj_ep{e}.pt",map_location="cpu"); state=d.get("proj",d); state={k.removeprefix("net."):v for k,v in state.items()}; p=core.build_proj();p.load_state_dict(state);p.eval().to(device);out[f"ep{e:02d}"]=core.embedding_from_features(raw,p)
    for seed in range(1,11): torch.manual_seed(seed); out[f"z0_s{seed}"]=core.embedding_from_features(raw,core.build_proj().eval().to(device))
    return out
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--pool",type=Path,required=True); ap.add_argument("--backbone",type=Path,required=True); ap.add_argument("--proj-dir",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); ap.add_argument("--feat-cache",type=Path); ap.add_argument("--device",choices=("cpu","cuda"),default="cpu"); ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--ratios",nargs="+",type=float,default=[.005,.01,.02]); ap.add_argument("--z0-seeds",nargs="+",type=int,default=list(range(1,11))); a=ap.parse_args(argv)
    manifest=json.loads(a.pool.read_text(encoding="utf-8"))
    try: image_paths,paths=load_paths(manifest)
    except ValueError as exc: raise SystemExit(str(exc))
    if not exact_epochs(a.proj_dir): raise SystemExit("requires exact proj_ep1..20 coverage")
    dials=ratio_dials(len(paths),a.ratios)
    if a.z0_seeds != list(range(1,11)): raise SystemExit("z0 seeds must be preregistered 1..10")
    if a.device=="cuda":
        import torch
        if not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
        raw=os.environ.get("REPRO_GPU_MEMORY_FRACTION")
        if raw is not None: torch.cuda.set_per_process_memory_fraction(float(raw),device=0)
    emb=embeddings(a.backbone,a.proj_dir,image_paths,a.feat_cache,a.device,a.batch_size)
    per=[]; predictions={}
    for dial in dials:
        rows={}
        for name,z in emb.items():
            metric,pred=label_free_metrics(z,dial); rows[name]=metric; predictions[f"{dial['mcs']}_{dial['ms']}_{name}"]=pred
        rule_c=rule_c_select(rows)
        per.append({"dial":dial,"selected_ep":rule_c["selected_ep"],"metrics":rows,"rule_c":rule_c})
    sensitivity={"mcs":6,"ms":3,"method":"leaf","eps":.06}
    sensitivity_rows={}
    for name,z in emb.items():
        metric,pred=label_free_metrics(z,sensitivity); sensitivity_rows[name]=metric; predictions[f"6_3_{name}"]=pred
    selected=consensus(per)
    a.out_dir.mkdir(parents=True,exist_ok=True); npz=a.out_dir/"prediction_embedding_bundle.npz"; tmp=npz.with_suffix(".tmp.npz"); np.savez_compressed(tmp,paths=np.asarray(paths),**emb,**predictions); os.replace(tmp,npz)
    model_names=["frozen",*[f"ep{i:02d}" for i in range(1,21)],*[f"z0_s{i}" for i in range(1,11)]]
    bundle={"pool_sha256":sha(a.pool),"root":manifest["root"],"paths":paths,"npz_path":str(npz.resolve()),"npz_sha256":sha(npz),"checkpoint_sha256":{p.name:sha(p) for p in sorted(a.proj_dir.glob("proj_ep*.pt"))},"z0_seeds":a.z0_seeds,"models":model_names}
    bundle_path=a.out_dir/"prediction_embedding_bundle.json"; bundle_sha=atomic(bundle_path,bundle)
    if selected is None:
        reason="no 2/3 label-free epoch consensus"
        snap={"schema_version":"rule_c_label_free_selection.v2","status":"diagnostic_no_consensus","selection_valid":False,"reason":reason,"selection_policy":{"name":"gate_pass_k_q75_min_pre_reassign_noise","k_percentile":75,"noise_tie_break":"lowest_epoch"},"pool":str(a.pool.resolve()),"pool_sha256":sha(a.pool),"primary_dials":dials,"sensitivity_audit":{"dial":sensitivity,"metrics":sensitivity_rows,"consensus_eligible":False},"per_dial":per,"consensus_selected_ep":None,"bundle_path":str(bundle_path.resolve()),"bundle_sha256":bundle_sha,"z0_seeds":a.z0_seeds,"offline_labels_evaluated":False}
        snap_path=a.out_dir/"selection_snapshot.json"; snap_sha=atomic(snap_path,snap)
        print(json.dumps({"selection_snapshot_path":str(snap_path.resolve()),"selection_snapshot_sha256":snap_sha,"status":snap["status"]}))
        raise SystemExit(reason)
    snap={"schema_version":"rule_c_label_free_selection.v2","status":"selected","selection_valid":True,"selection_policy":{"name":"gate_pass_k_q75_min_pre_reassign_noise","k_percentile":75,"noise_tie_break":"lowest_epoch"},"pool":str(a.pool.resolve()),"pool_sha256":sha(a.pool),"primary_dials":dials,"sensitivity_audit":{"dial":sensitivity,"metrics":sensitivity_rows,"consensus_eligible":False},"per_dial":per,"consensus_selected_ep":selected,"bundle_path":str(bundle_path.resolve()),"bundle_sha256":bundle_sha,"z0_seeds":a.z0_seeds,"offline_labels_evaluated":False}
    snap_path=a.out_dir/"selection_snapshot.json"; print(json.dumps({"selection_snapshot_path":str(snap_path.resolve()),"selection_snapshot_sha256":atomic(snap_path,snap)}))
if __name__=="__main__": main()
