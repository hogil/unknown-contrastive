"""Offline-only evaluator; cannot choose checkpoints or dials."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, uuid, re, numpy as np
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def path_order_sha(paths): return hashlib.sha256(json.dumps(paths,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()
V3_SCHEMA="rule_c_label_free_selection.v3"; V3_NAME="primary_all_support_minimax_pre_reassign_noise_v3"; SEAL_SCHEMA="rule_c_epoch_seal.v3"
def reject(message): raise SystemExit(message)
def validate_v3_before_labels(snapshot_path, expected_snapshot_sha, seal_path, expected_seal_sha):
 if not snapshot_path.is_file() or sha(snapshot_path)!=expected_snapshot_sha: reject("selection snapshot missing or hash mismatch")
 if not seal_path.is_file() or sha(seal_path)!=expected_seal_sha: reject("epoch seal missing or hash mismatch")
 try: s=json.loads(snapshot_path.read_text(encoding="utf-8")); seal=json.loads(seal_path.read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError): reject("invalid v3 snapshot or epoch seal")
 required=("source_v2_snapshot_sha256","r3_binding_sha256","selector_source_path","selector_source_sha256","source_bundle","selected_checkpoint")
 if (s.get("schema_version")!=V3_SCHEMA or s.get("selection_name")!=V3_NAME or s.get("status")!="selected" or s.get("selection_valid") is not True or s.get("labels_used") is not False or s.get("offline_labels_evaluated") is not False or not isinstance(s.get("selected_epoch"),int) or any(not s.get(k) for k in required)): reject("invalid or legacy v3 selection snapshot")
 bundle=s["source_bundle"]; checkpoint=s["selected_checkpoint"]
 if not all(isinstance(bundle.get(k),str) and bundle.get(k) for k in ("bundle_path","bundle_sha256","npz_path","npz_sha256")): reject("v3 bundle provenance missing")
 if not all(isinstance(checkpoint.get(k),str) and checkpoint.get(k) for k in ("selected_checkpoint_path","selected_checkpoint_sha256")): reject("v3 checkpoint provenance missing")
 if (not Path(bundle["bundle_path"]).is_file() or sha(bundle["bundle_path"])!=bundle["bundle_sha256"] or not Path(bundle["npz_path"]).is_file() or sha(bundle["npz_path"])!=bundle["npz_sha256"] or not Path(checkpoint["selected_checkpoint_path"]).is_file() or sha(checkpoint["selected_checkpoint_path"])!=checkpoint["selected_checkpoint_sha256"]): reject("v3 bound cache or checkpoint hash mismatch")
 source=Path(s.get("source_v2_snapshot_path", "")); r3=Path(s.get("r3_binding_path", ""))
 selector=Path(s.get("selector_source_path", ""))
 if (not source.is_file() or sha(source)!=s["source_v2_snapshot_sha256"] or not r3.is_file() or sha(r3)!=s["r3_binding_sha256"] or not selector.is_file() or sha(selector)!=s["selector_source_sha256"]): reject("v3 source or R3 binding hash mismatch")
 try: source_payload=json.loads(source.read_text(encoding="utf-8")); r3_payload=json.loads(r3.read_text(encoding="utf-8")); pool=json.loads(Path(bundle.get("pool_path", s.get("pool", ""))).read_text(encoding="utf-8")); meta=json.loads(Path(bundle["bundle_path"]).read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError): reject("v3 source provenance invalid")
 repair=r3_payload.get("response",{}).get("rule_c_repair",{}); spec=repair.get("specification",{}) if isinstance(repair,dict) else {}
 if (source_payload.get("schema_version")!="rule_c_label_free_selection.v2" or r3_payload.get("evidence_packet_sha")!=s.get("r3_evidence_packet_sha256") or r3_payload.get("binding_sha256")!=s.get("r3_decision_binding_sha256") or repair.get("decision")!="implement_exact_v3" or spec.get("selection_name")!=V3_NAME or repair.get("selected_epoch_on_frozen_snapshot")!=14 or repair.get("labels_used") is not False or repair.get("code_only_implementation_authorized") is not True): reject("v3 source or R3 binding semantics mismatch")
 paths=pool.get("files")
 if (not isinstance(paths,list) or not all(isinstance(x,str) for x in paths) or sha(Path(bundle.get("pool_path", s.get("pool", ""))))!=s.get("pool_sha256") or len(paths)!=s.get("selection_manifest_count") or path_order_sha(paths)!=bundle.get("ordered_paths_sha256") or meta.get("paths")!=paths): reject("v3 bound pool path order mismatch")
 with np.load(bundle["npz_path"],allow_pickle=False) as bound_npz:
  if bound_npz["paths"].tolist()!=paths: reject("v3 bound NPZ path order mismatch")
 if (seal.get("schema_version")!=SEAL_SCHEMA or seal.get("selection_name")!=V3_NAME or seal.get("labels_used") is not False or seal.get("selection_snapshot_sha256")!=expected_snapshot_sha or seal.get("selected_epoch")!=s["selected_epoch"] or any(seal.get(k)!=s.get(k) for k in ("source_v2_snapshot_sha256","r3_binding_sha256","selector_source_sha256")) or seal.get("bundle_sha256")!=bundle["bundle_sha256"] or seal.get("npz_sha256")!=bundle["npz_sha256"] or seal.get("selected_checkpoint_sha256")!=checkpoint["selected_checkpoint_sha256"]): reject("epoch seal snapshot binding mismatch")
 if any(seal.get(k)!=bundle.get(k) for k in ("pool_path","pool_sha256","pool_count","ordered_paths_sha256")): reject("epoch seal pool binding mismatch")
 return s,Path(bundle["bundle_path"]),meta
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument("--selection-snapshot",type=Path,required=True); ap.add_argument("--expected-selection-sha256",required=True); ap.add_argument("--epoch-seal",type=Path,required=True); ap.add_argument("--expected-epoch-seal-sha256",required=True); ap.add_argument("--labeled-pool",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(argv)
 s,b,meta=validate_v3_before_labels(a.selection_snapshot,a.expected_selection_sha256,a.epoch_seal,a.expected_epoch_seal_sha256)
 m=json.loads(a.labeled_pool.read_text(encoding="utf-8")); files=m.get("files",[])
 if not files or not all(isinstance(x,dict) and isinstance(x.get("path"),str) and "label" in x for x in files): raise SystemExit("labeled manifest required for offline audit")
 if m.get("root") != meta.get("root"): raise SystemExit("manifest root mismatch")
 if sha(meta["npz_path"])!=meta.get("npz_sha256") or meta.get("paths")!=[x["path"] for x in files]: raise SystemExit("bundle hash or path order mismatch")
 from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
 labels=np.asarray([x["label"] for x in files]); out={}
 # Pairing is deterministic and uses manifest provenance before a filename token.
 block_ids=[]; block_methods=[]; fallback=0
 for item in files:
  explicit=next(((k,item.get(k)) for k in ("block_id","lot","source","temporal_batch","group") if item.get(k) not in (None,"")),None)
  value=explicit[1] if explicit else None
  if value is None:
   value=Path(item["path"]).stem.split("_")[0]; fallback+=1; block_methods.append("fallback:filename_token_v1")
  else: block_methods.append(f"explicit:{explicit[0]}")
  block_ids.append(str(value))
 block_sizes=Counter(block_ids); method_counts=Counter(block_methods)
 block_provenance={"method":"block_id/lot/source/temporal_batch/group; filename stem first token fallback",
  "method_counts":dict(sorted(method_counts.items())),"fallback_count":fallback,
  "explicit_count":len(files)-fallback,"n_entries":len(files),"n_unique_blocks":len(block_sizes),
  "singleton_blocks":sum(v==1 for v in block_sizes.values()),"max_block_size":max(block_sizes.values())}
 with np.load(meta["npz_path"],allow_pickle=False) as npz:
  if list(npz["paths"])!=[x["path"] for x in files]: raise SystemExit("NPZ path order mismatch")
  for dialrow in s["per_dial"]:
   d=dialrow["dial"]; key=f"{d['mcs']}_{d['ms']}_"; models=["frozen",f"ep{s['selected_epoch']:02d}",* [f"z0_s{i}" for i in range(1,11)]]
   rows={}
   for name in models:
    pred=npz[key+name]; classes=[]; reps=[]
    from scripts.cluster_metrics import capture_metrics
    cap=capture_metrics(pred,labels); dominant=cap["dominant_by_cluster"]
    for lab in np.unique(labels):
     ix=labels==lab; good=[c for c,v in dominant.items() if v==lab]; value=float(np.mean(np.isin(pred[ix],good))); classes.append((str(lab),value)); reps.append(good[0] if good else -1)
    from scripts.eval_open_set_embeddings import purity_score
    from scripts.eval_open_set_embeddings import reassign_noise_to_nearest_cluster
    post, post_info = reassign_noise_to_nearest_cluster(npz[name], pred, "nearest_q90")
    block_rows=[]
    for block in sorted(set(block_ids)):
     bix=np.asarray([x==block for x in block_ids]); block_labels=labels[bix]; block_pred=pred[bix]
     values=[]
     for lab in np.unique(block_labels):
      good=[c for c,v in dominant.items() if v==lab]
      values.append(float(np.mean(np.isin(block_pred[block_labels==lab],good))))
     block_rows.append({"block_id":block,"n":int(bix.sum()),"image_cap":float(np.mean(values)),"pre_noise":float(np.mean(block_pred==-1)*100)})
    rows[name]={
    "P1_unique_dominant_capture":cap["capture_rate"], "captured_classes":cap["captured_classes"], "lost_classes":sorted(set(np.unique(labels).tolist())-set(cap["captured_classes"]),key=str), "macro_image_cap":float(np.mean([v for _,v in classes])),
    "minimum_image_cap":float(np.min([v for _,v in classes])), "per_class_image_cap":classes,
    "pre_reassign_noise":float(np.mean(pred==-1)*100), "k":int(len(set(pred.tolist()))-(1 if -1 in pred else 0)),
    "post_reassign_noise":float(np.mean(post==-1)*100), "noise_reassigned":post_info["noise_reassigned"], "primary_metrics_use":"pre_reassign_predictions",
    "purity_weighted":float(purity_score(labels,pred)),
    "fragmentation":float(len({int(c) for c in pred if int(c)>=0}) / max(1, len(set(labels.tolist())))),
    "ARI":float(adjusted_rand_score(labels,pred)), "AMI":float(adjusted_mutual_info_score(labels,pred)),
    "representative_cluster_ids":{str(lab): int(cluster) for lab, cluster in zip(np.unique(labels), reps)},
    "block_metrics":block_rows,"block_provenance":block_provenance}
   out[str(d["ratio"])]=rows
 a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_name("."+a.out.name+"."+uuid.uuid4().hex); tmp.write_text(json.dumps({
  "selection_snapshot_path":str(a.selection_snapshot.resolve()),"selection_snapshot_sha256":a.expected_selection_sha256,
  "unlabeled_pool_path":s.get("pool"),"unlabeled_pool_sha256":s.get("pool_sha256"),
  "labeled_pool_path":str(a.labeled_pool.resolve()),"labeled_pool_sha256":sha(a.labeled_pool),
  "bundle_path":str(b.resolve()),"bundle_sha256":s["source_bundle"]["bundle_sha256"],
  "npz_path":str(Path(meta["npz_path"]).resolve()),"npz_sha256":s["source_bundle"]["npz_sha256"],
  "manifest_root":m.get("root"),"selected_ep":s["selected_epoch"],"epoch_seal_path":str(a.epoch_seal.resolve()),"epoch_seal_sha256":a.expected_epoch_seal_sha256,
  "primary_dials":s["primary_dials"],"offline_only":True,"block_provenance":block_provenance,
  "metrics":out},indent=2),encoding="utf-8"); os.replace(tmp,a.out)
if __name__=="__main__": main()
