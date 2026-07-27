import hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_rule_c_screen import evaluate,main
from scripts.run_rule_c_selector import rule_c_select

R=(.005,.01,.02)
def selector(noise=10, selected=2, third_selected=None):
 def metrics(noise):
  rows={f"ep{i:02d}":{"k":10,"pre_reassign_noise":noise+100,"over_merge":1,"stability":0,"coherence":0} for i in range(1,21)}
  rows["ep02"]={"k":10,"pre_reassign_noise":noise,"over_merge":0,"stability":.75,"coherence":.80}
  return rows
 per=[]
 for i,r in enumerate(R):
  own=third_selected if i==2 and third_selected else selected
  rows=metrics(noise+i)
  if own != 2:
   rows["ep02"]={"k":10,"pre_reassign_noise":noise+i+1,"over_merge":0,"stability":.75,"coherence":.80}
   rows[f"ep{own:02d}"]={"k":10,"pre_reassign_noise":noise+i,"over_merge":0,"stability":.75,"coherence":.80}
  rule=rule_c_select(rows); per.append({"dial":{"ratio":r},"selected_ep":rule["selected_ep"],"metrics":rows,"rule_c":rule})
 return {"schema_version":"rule_c_label_free_selection.v2","selection_policy":{"name":"gate_pass_k_q75_min_pre_reassign_noise","k_percentile":75,"noise_tie_break":"lowest_epoch"},"pool":"pool.json","pool_sha256":"pool-sha","bundle_path":"bundle.json","bundle_sha256":"bundle-sha","primary_dials":[{"ratio":r} for r in R],"consensus_selected_ep":selected,"per_dial":per}
def offline(add=0,noise=10,blocks=("L1","L2"), z0_fail_ratio=None):
 metrics={}
 for r in R:
  models={}
  for name in ["frozen","ep02",*[f"z0_s{i}" for i in range(1,11)]]:
   z=name.startswith("z0_"); cap=2 if z else 2+add; n=20 if z else noise
   if not z and r==z0_fail_ratio: cap,n=2,20
   models[name]={"P1_unique_dominant_capture":1,"captured_classes":["A","B"],"macro_image_cap":cap,"minimum_image_cap":cap,"per_class_image_cap":[["A",cap],["B",cap]],"pre_reassign_noise":n,"post_reassign_noise":n,"block_metrics":[{"block_id":x,"n":2,"image_cap":cap,"pre_noise":n} for x in blocks],"block_provenance":{"fallback_count":0,"explicit_count":len(blocks),"n_entries":len(blocks),"n_unique_blocks":len(blocks)}}
  metrics[str(r)]=models
 return {"selected_ep":2,"unlabeled_pool_path":"pool.json","unlabeled_pool_sha256":"pool-sha","bundle_path":"bundle.json","bundle_sha256":"bundle-sha","metrics":metrics}
def test_real_schema_passes_and_is_nonpromoting():
 blocks=tuple(f"L{i}" for i in range(20));out=evaluate(selector(20,third_selected=4),selector(10,third_selected=7),offline(blocks=blocks),offline(1,10,blocks=blocks,z0_fail_ratio=.02),iterations=100)
 assert all(out["gates"].values()) and out["promotes"] is False
 assert out["paired_bootstrap"]["n_blocks"]==20 and out["z0_pass_count"]==2
def test_consensus_capture_z0_and_block_fail_closed():
 bad=selector()
 for row in bad["per_dial"][:2]:
  row["metrics"]["ep02"]={"k":10,"pre_reassign_noise":11,"over_merge":0,"stability":.75,"coherence":.80}
  row["metrics"]["ep03"]={"k":10,"pre_reassign_noise":10,"over_merge":0,"stability":.75,"coherence":.80}
  row["rule_c"]=rule_c_select(row["metrics"]);row["selected_ep"]=row["rule_c"]["selected_ep"]
 try:evaluate(bad,selector(),offline(),offline(),iterations=10)
 except ValueError as e:assert "consensus" in str(e)
 else:assert False
 lr=offline(1,10);lr["metrics"]["0.005"]["ep02"]["captured_classes"]=["A"]
 assert not evaluate(selector(20),selector(10),offline(),lr,iterations=10)["gates"]["captured_class_plus_macro_min_image_cap_nonregression"]
 macro_fail=offline(1,10); row=macro_fail["metrics"]["0.005"]["ep02"]; row["macro_image_cap"]=1; row["minimum_image_cap"]=1
 assert not evaluate(selector(20),selector(10),offline(),macro_fail,iterations=10)["gates"]["captured_class_plus_macro_min_image_cap_nonregression"]
 weak=offline(0,20)
 assert not evaluate(selector(20),selector(10),offline(),weak,iterations=10)["gates"]["z0_capacity_matched_primary_gate_2_of_3"]
 mismatch=offline(1,10,blocks=("OTHER",))
 try:evaluate(selector(20),selector(10),offline(),mismatch,iterations=10)
 except ValueError as e:assert "identities" in str(e)
 else:assert False
def test_sensitivity_excluded_and_main_hashes(tmp_path):
 bad={**selector(),"per_dial":[{"dial":{"ratio":.03},"selected_ep":2,"metrics":{"ep02":{"pre_reassign_noise":1}}}]}
 try:evaluate(bad,bad,offline(),offline(),iterations=10)
 except ValueError as e:assert "primary ratio" in str(e)
 else:assert False

 pool=tmp_path/"pool.json";bundle=tmp_path/"bundle.json";npz=tmp_path/"bundle.npz"
 for p in (pool,bundle,npz):p.write_text(p.name)
 files={"base_selector":selector(20),"lr_selector":selector(10),"base_offline":offline(blocks=tuple(range(20))),"lr_offline":offline(1,10,blocks=tuple(range(20)))};paths={}
 for prefix in ("base","lr"):
  sel,off=files[f"{prefix}_selector"],files[f"{prefix}_offline"]
  sel.update({"pool":str(pool),"pool_sha256":hashlib.sha256(pool.read_bytes()).hexdigest(),"bundle_path":str(bundle),"bundle_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest()})
  off.update({"unlabeled_pool_path":str(pool),"unlabeled_pool_sha256":sel["pool_sha256"],"bundle_path":str(bundle),"bundle_sha256":sel["bundle_sha256"],"labeled_pool_path":str(pool),"labeled_pool_sha256":sel["pool_sha256"],"npz_path":str(npz),"npz_sha256":hashlib.sha256(npz.read_bytes()).hexdigest()})
  p=tmp_path/(prefix+"_selector.json");p.write_text(json.dumps(sel));paths[f"{prefix}_selector"]=p
  off["selection_snapshot_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
  p=tmp_path/(prefix+"_offline.json");p.write_text(json.dumps(off));paths[f"{prefix}_offline"]=p
 expected={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}; ep=tmp_path/"sha.json";ep.write_text(json.dumps(expected));out=tmp_path/"out.json"
 assert main(["--base-selector",str(paths["base_selector"]),"--lr-selector",str(paths["lr_selector"]),"--base-offline",str(paths["base_offline"]),"--lr-offline",str(paths["lr_offline"]),"--expected-shas",str(ep),"--out",str(out),"--bootstrap-iterations","10"])==0
 assert json.loads(out.read_text())["input_sha256"]==expected

def test_screen_rejects_selector_that_skips_rule_c_k_retention():
 bad=selector()
 row=bad["per_dial"][0]; rows=row["metrics"]
 rows["ep01"]={"k":1,"pre_reassign_noise":0,"over_merge":0,"stability":.75,"coherence":.80}
 rows["ep02"]={"k":10,"pre_reassign_noise":1,"over_merge":0,"stability":.75,"coherence":.80}
 rows["ep03"]={"k":10,"pre_reassign_noise":2,"over_merge":0,"stability":.75,"coherence":.80}
 rows["ep04"]={"k":10,"pre_reassign_noise":3,"over_merge":0,"stability":.75,"coherence":.80}
 row["selected_ep"]=1
 row["rule_c"]={"gate_pass_epochs":[1,2,3,4],"k_q75":10.0,"retained_epochs":[1,2,3,4],"selected_ep":1,"tie_break":"lowest_epoch"}
 with pytest.raises(ValueError,match="Rule-C selected epoch mismatch"):
  evaluate(bad,selector(10),offline(),offline(),iterations=10)

@pytest.mark.parametrize("provenance",[{"fallback_count":1,"explicit_count":0,"n_entries":1,"n_unique_blocks":1},{"fallback_count":0,"explicit_count":1,"n_entries":1,"n_unique_blocks":1}])
def test_fallback_or_singleton_blocks_are_descriptive_only(provenance):
 b,l=offline(blocks=("one",)),offline(1,10,blocks=("one",))
 for output in (b,l):
  for rows in output["metrics"].values():
   for row in rows.values():row["block_provenance"]=provenance
 result=evaluate(selector(20),selector(10),b,l,iterations=10)
 assert result["gates"]["explicit_block_provenance_minimum"] is False and result["paired_bootstrap"]["inferential"] is False

@pytest.mark.parametrize("target,needle",[("snapshot","selection-snapshot SHA"),("pool","unlabeled-pool"),("bundle","selector-bundle")])
def test_screen_rejects_offline_selector_pool_or_bundle_binding_drift(target,needle):
 b,l=offline(blocks=tuple(range(20))),offline(1,10,blocks=tuple(range(20)))
 if target=="snapshot":
  b["selection_snapshot_sha256"]="wrong"; l["selection_snapshot_sha256"]="wrong"
  kwargs={"base_selector_sha":"expected","lr_selector_sha":"expected"}
 elif target=="pool": b["unlabeled_pool_sha256"]="wrong";kwargs={}
 else: b["bundle_sha256"]="wrong";kwargs={}
 with pytest.raises(ValueError,match=needle):evaluate(selector(20),selector(10),b,l,iterations=10,**kwargs)

def test_screen_rejects_offline_provenance_file_drift(tmp_path):
 pool,bundle,npz=(tmp_path/"pool.json",tmp_path/"bundle.json",tmp_path/"bundle.npz")
 for path in (pool,bundle,npz):path.write_text(path.name)
 b,l=offline(blocks=tuple(range(20))),offline(1,10,blocks=tuple(range(20)))
 bs,ls=selector(20),selector(10)
 for sel,out in ((bs,b),(ls,l)):
  sel.update({"pool":str(pool),"pool_sha256":hashlib.sha256(pool.read_bytes()).hexdigest(),"bundle_path":str(bundle),"bundle_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest()})
  out.update({"unlabeled_pool_path":str(pool),"unlabeled_pool_sha256":sel["pool_sha256"],"labeled_pool_path":str(pool),"labeled_pool_sha256":sel["pool_sha256"],"bundle_path":str(bundle),"bundle_sha256":sel["bundle_sha256"],"npz_path":str(npz),"npz_sha256":hashlib.sha256(npz.read_bytes()).hexdigest()})
 bundle.write_text("tampered")
 with pytest.raises(ValueError,match="file drift"):evaluate(bs,ls,b,l,iterations=10,verify_files=True)
