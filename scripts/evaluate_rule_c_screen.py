#!/usr/bin/env python3
"""Read-only, CPU-only Rule-C base-vs-LR.008 screen; it never promotes or launches."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile
from pathlib import Path
from scripts.run_rule_c_selector import rule_c_select

PRIMARY_RATIOS=(.005,.01,.02)
def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 v=json.loads(Path(p).read_text(encoding="utf-8"));
 if not isinstance(v,dict): raise ValueError("JSON object required")
 return v
def atomic_json(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",delete=False,dir=p.parent) as f: json.dump(v,f,indent=2,sort_keys=True); t=f.name
 os.replace(t,p)
def q(values,p):
 values=sorted(map(float,values)); return values[min(len(values)-1, max(0, math.ceil(p*len(values))-1))]
def selector_rows(s):
 if s.get("schema_version")!="rule_c_label_free_selection.v2": raise ValueError("Rule-C selector snapshot v2 required")
 if s.get("selection_policy")!={"name":"gate_pass_k_q75_min_pre_reassign_noise","k_percentile":75,"noise_tie_break":"lowest_epoch"}: raise ValueError("Rule-C selector policy mismatch")
 if not isinstance(s.get("consensus_selected_ep"),int): raise ValueError("global consensus_selected_ep required")
 rows={}
 for row in s.get("per_dial",[]):
  dial=row.get("dial",{}); ratio=float(dial.get("ratio"))
  if ratio in PRIMARY_RATIOS:
   metrics=row.get("metrics",{}); name=f"ep{s['consensus_selected_ep']:02d}"
   if name not in metrics or "pre_reassign_noise" not in metrics[name]: raise ValueError(f"selected label-free metric missing at {ratio}")
   recomputed=rule_c_select(metrics)
   if row.get("selected_ep")!=recomputed["selected_ep"] or row.get("rule_c")!=recomputed: raise ValueError(f"Rule-C selected epoch mismatch at {ratio}")
   rows[ratio]=metrics[name]
 if set(rows)!=set(PRIMARY_RATIOS): raise ValueError("exact primary ratio dials .005/.01/.02 required; sensitivity excluded")
 choices=[x.get("selected_ep") for x in s.get("per_dial",[]) if float(x.get("dial",{}).get("ratio",-1)) in PRIMARY_RATIOS]
 if choices.count(s["consensus_selected_ep"])<2: raise ValueError("global consensus epoch is not selected by 2/3 primary dials")
 return rows
def verify_offline_provenance(selector,offline,selector_sha=None,verify_files=False):
 if offline.get("selected_ep")!=selector.get("consensus_selected_ep"): raise ValueError("offline selected epoch differs from selector consensus")
 if offline.get("unlabeled_pool_path")!=selector.get("pool") or offline.get("unlabeled_pool_sha256")!=selector.get("pool_sha256"): raise ValueError("offline unlabeled-pool provenance mismatch")
 if offline.get("bundle_path")!=selector.get("bundle_path") or offline.get("bundle_sha256")!=selector.get("bundle_sha256"): raise ValueError("offline selector-bundle provenance mismatch")
 if selector_sha is not None and offline.get("selection_snapshot_sha256")!=selector_sha: raise ValueError("offline selection-snapshot SHA mismatch")
 if verify_files:
  for path_key,sha_key in (("labeled_pool_path","labeled_pool_sha256"),("bundle_path","bundle_sha256"),("npz_path","npz_sha256")):
   path=Path(str(offline.get(path_key,""))); expected=offline.get(sha_key)
   if not path.is_file() or not expected or sha256_file(path)!=expected: raise ValueError(f"offline provenance file drift: {path_key}")
def offline_row(o,ratio,model):
 metrics=o.get("metrics",{}); rows=next((v for k,v in metrics.items() if float(k)==ratio),None)
 if not isinstance(rows,dict) or model not in rows: raise ValueError(f"offline selected model missing at {ratio}")
 return rows[model],rows
def blocks(row):
 result={str(x["block_id"]):x for x in row.get("block_metrics",[]) if "block_id" in x}
 if not result: raise ValueError("selected-model block_metrics missing")
 return result
def ci(values,seed,iters):
 if not values or iters <= 0: raise ValueError("paired bootstrap requires blocks and positive iterations")
 rng=random.Random(seed); n=len(values); samples=sorted(sum(rng.choice(values) for _ in range(n))/n for _ in range(iters))
 return [samples[math.floor(.025*iters)],samples[math.ceil(.975*iters)-1]]
def evaluate(bs,ls,bo,lo,seed=42,iterations=2000,base_selector_sha=None,lr_selector_sha=None,min_explicit_blocks=20,verify_files=False):
 verify_offline_provenance(bs,bo,base_selector_sha,verify_files); verify_offline_provenance(ls,lo,lr_selector_sha,verify_files)
 bsel,lsel=selector_rows(bs),selector_rows(ls); wins=[]; details={}; caps_ok=True; z0_passes=0; block_deltas={}; explicit_blocks_ok=True; provenance={}
 bmodel=f"ep{bs['consensus_selected_ep']:02d}"; lmodel=f"ep{ls['consensus_selected_ep']:02d}"
 for r in PRIMARY_RATIOS:
  wins.append(float(lsel[r]["pre_reassign_noise"]) < float(bsel[r]["pre_reassign_noise"]))
  b,br=offline_row(bo,r,bmodel); l,lr=offline_row(lo,r,lmodel)
  bc=dict(b["per_class_image_cap"]); lc=dict(l["per_class_image_cap"])
  lost=sorted(set(b["captured_classes"])-set(l["captured_classes"]),key=str); reg=[x for x in set(bc)|set(lc) if float(lc.get(x,0))<float(bc.get(x,0))]
  # Individual regressions are evidence for R3, not an unapproved all-class gate.
  capok=not lost and float(l["macro_image_cap"])>=float(b["macro_image_cap"]) and float(l["minimum_image_cap"])>=float(b["minimum_image_cap"])
  caps_ok &= capok
  zkeys={f"z0_s{i}" for i in range(1,11)}
  if not zkeys.issubset(lr) or {k for k in lr if k.startswith("z0_s")} != zkeys: raise ValueError(f"requires exact z0_s1..10 at {r}")
  z=[lr[f"z0_s{i}"] for i in range(1,11)]
  zcap=[x["macro_image_cap"] for x in z]; znoise=[x["pre_reassign_noise"] for x in z]
  capwin=float(l["macro_image_cap"])>q(zcap,.95); noisewin=float(l["pre_reassign_noise"])<q(znoise,.05)
  zpass=(capwin and float(l["pre_reassign_noise"])<=q(znoise,.50)) or (noisewin and float(l["macro_image_cap"])>=q(zcap,.50)); z0_passes += int(zpass)
  bb,ll=blocks(b),blocks(l)
  if set(bb)!=set(ll): raise ValueError(f"paired block identities differ at {r}")
  bp,lp=b.get("block_provenance",{}),l.get("block_provenance",{})
  provenance[str(r)]={"base":bp,"lr008":lp}
  explicit_blocks_ok &= (bp.get("fallback_count")==0 and lp.get("fallback_count")==0
   and bp.get("explicit_count")==bp.get("n_entries") and lp.get("explicit_count")==lp.get("n_entries")
   and int(bp.get("n_unique_blocks",0))>=int(min_explicit_blocks)
   and int(lp.get("n_unique_blocks",0))>=int(min_explicit_blocks))
  for k in sorted(bb):
   entry=block_deltas.setdefault(k,{"cap":[],"noise":[]}); entry["cap"].append(float(ll[k]["image_cap"])-float(bb[k]["image_cap"])); entry["noise"].append(float(bb[k]["pre_noise"])-float(ll[k]["pre_noise"]))
  details[str(r)]={"capture_lost":lost,"cap_regressions":reg,"z0":{"cap_q95":q(zcap,.95),"noise_q05":q(znoise,.05),"cap_win":capwin,"noise_win":noisewin,"pass":zpass}}
 if any(len(v["cap"])!=3 or len(v["noise"])!=3 for v in block_deltas.values()): raise ValueError("block coverage must be complete across all primary dials")
 bootstrap_cap=[sum(v["cap"])/3 for v in block_deltas.values()]; bootstrap_noise=[sum(v["noise"])/3 for v in block_deltas.values()]
 capci,noiseci=ci(bootstrap_cap,seed,iterations),ci(bootstrap_noise,seed,iterations)
 bootok=capci[0]>0 or noiseci[0]>0
 gates={"captured_class_plus_macro_min_image_cap_nonregression":caps_ok,"label_free_pre_noise_wins_2_of_3":sum(wins)>=2,"z0_capacity_matched_primary_gate_2_of_3":z0_passes>=2,"explicit_block_provenance_minimum":explicit_blocks_ok,"paired_block_ci_improves_image_cap_or_noise":bootok and explicit_blocks_ok}
 return {"schema_version":"rule_c_paired_screen.v3","promotes":False,"primary_ratios":list(PRIMARY_RATIOS),"gates":gates,"reasons":[k for k,v in gates.items() if not v],"label_free_noise_wins":sum(wins),"per_ratio":details,"block_provenance":provenance,"minimum_explicit_blocks":min_explicit_blocks,"paired_bootstrap":{"inferential":explicit_blocks_ok,"seed":seed,"iterations":iterations,"n_blocks":len(block_deltas),"image_cap_ci95":capci,"pre_noise_improvement_ci95":noiseci,"rule":"strict lower CI > 0 for image-cap increase OR pre-noise decrease; descriptive-only when explicit provenance is absent"},"z0_pass_count":z0_passes,"z0_rule":"Per dial: selected macro image-cap > z0 q95 while pre-noise is no higher than z0 median, OR selected pre-noise < z0 q05 while image-cap is no lower than z0 median; pass requires 2/3 primary dials."}
def main(argv=None):
 p=argparse.ArgumentParser()
 for n in ("base-selector","lr-selector","base-offline","lr-offline"): p.add_argument("--"+n,required=True)
 p.add_argument("--expected-shas",required=True);p.add_argument("--out",required=True);p.add_argument("--bootstrap-seed",type=int,default=42);p.add_argument("--bootstrap-iterations",type=int,default=2000);p.add_argument("--min-explicit-blocks",type=int,default=20);a=p.parse_args(argv)
 paths={"base_selector":a.base_selector,"lr_selector":a.lr_selector,"base_offline":a.base_offline,"lr_offline":a.lr_offline}; actual={k:sha256_file(v) for k,v in paths.items()}
 if actual!=load(a.expected_shas): raise SystemExit("immutable input SHA mismatch")
 result=evaluate(load(a.base_selector),load(a.lr_selector),load(a.base_offline),load(a.lr_offline),a.bootstrap_seed,a.bootstrap_iterations,actual["base_selector"],actual["lr_selector"],a.min_explicit_blocks,True);result["input_sha256"]=actual;atomic_json(a.out,result);return 0
if __name__=="__main__": raise SystemExit(main())
