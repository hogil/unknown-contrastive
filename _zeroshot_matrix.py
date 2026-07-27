#!/usr/bin/env python3
"""Zero-shot transfer matrix — 배포 우선순위 1순위 경로 검증.

학습 0, inference 만: 이미 학습된 proj head 를 처음 보는 pool 의 frozen FCMAE feature 에
그대로 적용 (재학습·재선택 없음) → 같은 pool 의 frozen baseline 과 채점 비교.

재사용(수정 없음): _grouping_eval.py 의 load_backbone / extract_backbone_features /
embedding_from_features / label_free / offline / build_proj / DIAL. 채점 로직 1비트도
안 바꿈 — 여기서 하는 일은 (pool × source) 조합을 돌며 같은 채점기를 호출하는 것뿐.

각 source 는 "최종(가장 많이 학습된) epoch" 고정 1개만 사용 — target pool 라벨/구조를
보고 epoch 을 재선택하지 않음 (순수 zero-shot: 배포되는 모델 = 학습 종료 시점 그 자체).
champion 계열(clean546 s42/s1/s2)은 애초에 checkpoint 폴더에 그 1개 epoch만 저장되어
있어 이 정의와 자동으로 일치.

feature 추출 device: --device (default cuda if available). ★ CPU/GPU 로 뽑은 raw feature 는
부동소수 비결정성(~1e-4)으로 HDBSCAN 경계 판정이 흔들릴 수 있어(실측 사례 있음) 섞지 않는다 —
feat_cache 재사용 시 저장된 meta 의 device 가 현재 --device 와 다르면 캐시를 버리고 재추출한다.
클러스터링(HDBSCAN 등)은 항상 CPU/numpy 라 이 이슈는 feature 추출 단계에만 해당.

출력: result_grouping/_zeroshot/matrix.csv, matrix.json, feat_cache/<pool>.npy(+meta)
"""
from __future__ import annotations
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _common import resolve_pool  # noqa: E402
import _grouping_eval as GE  # noqa: E402  (load_backbone/extract_backbone_features/embedding_from_features/label_free/offline/build_proj/DIAL)

GE.DIAL.update(mcs=6, ms=3)  # team-lead directive: pool 크기 차 감안, frozen/adapted 동일 다이얼로만 비교 (Δ 관심사)

OUT = REPO / "result_grouping" / "_zeroshot"
FEAT_CACHE = OUT / "feat_cache"
FEAT_CACHE.mkdir(parents=True, exist_ok=True)

POOLS = {  # id -> --pool arg (dir 또는 manifest json)
    "cca": "E:/data/images/cca",
    "hf_dtd": "E:/data/images/hf_dtd",
    "hf_resisc45": "E:/data/images/hf_resisc45",
    "hf_flowers102": "E:/data/images/hf_flowers102",
    "anchor_avg30": "data/pools/anchor_avg30_repro.json",
    "unknown_eval100": "data/pools/unknown_eval100.json",
    "dlrsd": "E:/data/images/dlrsd_extracted/DLRSD/Images",  # bonus (여유 있으면)
}

# name -> (ckpt path, 학습 도메인 id, coarse domain bucket)
DOMAIN_BUCKET = {  # coarse bucket for "domain distance" 분류
    "mwm38_clean546": "wafer_synth", "unknown_eval100": "wafer_synth",
    "anchor_avg30_repro": "wafer_synth", "unknown_pos150": "wafer_synth",
    "wm811k_defectonly": "wafer_real", "mixedwm38_all": "wafer_real", "mixedwm38_single7": "wafer_real",
    "cca": "wafer_real", "hf_dtd": "natural", "hf_resisc45": "natural", "hf_flowers102": "natural",
    "dlrsd": "natural",
}
POOL_DOMAIN = {  # target pool id -> 위 bucket key
    "cca": "cca", "hf_dtd": "hf_dtd", "hf_resisc45": "hf_resisc45", "hf_flowers102": "hf_flowers102",
    "anchor_avg30": "anchor_avg30_repro", "unknown_eval100": "unknown_eval100", "dlrsd": "dlrsd",
}

SOURCES = {
    "clean546_s42": ("runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt", "mwm38_clean546"),
    "clean546_s1": ("runs/sweep/abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt", "mwm38_clean546"),
    "clean546_s2": ("runs/sweep/abl_best_s2_B4_260724_111604/checkpoints/proj_ep17.pt", "mwm38_clean546"),
    "rehearsal_s42": ("runs/rehearsal_eval100/abl_rh_s42_B4_260725_195040/checkpoints/proj_ep9.pt", "unknown_eval100"),
    "rehearsal_s1": ("runs/rehearsal_eval100/abl_rh_s1_B4_260725_222815/checkpoints/proj_ep9.pt", "unknown_eval100"),
    "anchor_xds_rc": ("runs/anchor_xds/abl_rc260726_B4_260726_083547/checkpoints/proj_ep3.pt", "anchor_avg30_repro"),
    "wm811k": ("runs/wm811k/abl_wm_B4_260724_073901/checkpoints/proj_ep20.pt", "wm811k_defectonly"),
    "wm811k_fixed": ("runs/wm811k_fixed_b4/run/abl_wm811k_s42_B4_260724_074724/checkpoints/proj_ep20.pt", "wm811k_defectonly"),
    "mixedwm38": ("runs/mixedwm38/train_b4_260723_081721/checkpoints/proj_ep20.pt", "mixedwm38_all"),
    "single7": ("runs/single7/train_b4_260723_113356/checkpoints/proj_ep20.pt", "mixedwm38_single7"),
    "unknownpos": ("runs/unknownpos/abl_upos384_B4_260724_144250/checkpoints/proj_ep20.pt", "unknown_pos150"),
    "broaden_dtd": ("runs/broaden/abl_dtd_B4_260724_124440/checkpoints/proj_ep3.pt", "hf_dtd"),
    "broaden_resisc": ("runs/broaden/abl_resisc_B4_260724_123232/checkpoints/proj_ep15.pt", "hf_resisc45"),
    "broaden_anchorw": ("runs/broaden/abl_anchorw_B4_260724_130453/checkpoints/proj_ep20.pt", "anchor_avg30_repro"),
}


def build_adapter():
    return nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024))


class Head(nn.Module):
    def __init__(self, proj, adapt=None, gamma=None):
        super().__init__(); self.proj = proj; self.adapt = adapt
        self.register_buffer("gamma", gamma if gamma is not None else torch.zeros(1))

    def forward(self, x):
        if self.adapt is not None:
            x = x + self.gamma * self.adapt(x)
        return self.proj(x)


def load_head(ckpt_path):
    d = torch.load(ckpt_path, map_location="cpu")
    pj = d["proj"] if "proj" in d else d
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
    pr = GE.build_proj(); pr.load_state_dict(pj)
    ad = None
    if "adapt" in d:
        ad = build_adapter(); ad.load_state_dict(d["adapt"])
    return Head(pr, ad, d.get("gamma")).eval()


def get_pool_features(pool_id, pool_arg, bb, device):
    cache = FEAT_CACHE / f"{pool_id}.npy"
    meta_p = FEAT_CACHE / f"{pool_id}_meta.json"
    if cache.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("device") == device:
            feats = torch.from_numpy(np.load(cache)).float()
            print(f"[{pool_id}] cached features {feats.shape} (device={device}, matches)", flush=True)
            return feats, meta["labels"]
        print(f"[{pool_id}] cache device mismatch (cached={meta.get('device')} != requested={device}) "
              f"-> re-extracting to avoid CPU/GPU float nondeterminism mixing", flush=True)
    t0 = time.time()
    paths, labels = resolve_pool(pool_arg)
    print(f"[{pool_id}] {len(paths)} imgs, {len(set(labels))} classes - extracting raw GAP features ({device})...", flush=True)
    feats = GE.extract_backbone_features(paths, bb)
    np.save(cache, feats.numpy())
    meta_p.write_text(json.dumps({"paths": paths, "labels": labels, "device": device}, ensure_ascii=False), encoding="utf-8")
    print(f"[{pool_id}] done ({time.time()-t0:.0f}s) -> {cache}", flush=True)
    return feats, labels


def score(z, labels):
    lf = GE.label_free(z)
    off = GE.offline(z, labels, lf["pred"])
    return lf, off


def row_dict(pool_id, n, source, source_domain, epoch, device, lf, off):
    bucket_src = DOMAIN_BUCKET.get(source_domain, "?")
    bucket_tgt = DOMAIN_BUCKET.get(POOL_DOMAIN.get(pool_id, pool_id), "?")
    same_pool = (source_domain == POOL_DOMAIN.get(pool_id))
    relation = "SAME_POOL" if same_pool else ("same_bucket" if bucket_src == bucket_tgt else "cross_domain")
    return dict(pool=pool_id, n=n, source=source, source_domain=source_domain, epoch=epoch, device=device,
                domain_relation=relation, src_bucket=bucket_src, tgt_bucket=bucket_tgt,
                lf_k=lf["k"], lf_noise_pct=lf["noise_pct"], lf_non_noise_pct=lf["non_noise_pct"],
                lf_over_merge=lf["over_merge"], lf_stability=lf["stability"], lf_coherence=lf["coherence"],
                P1=off["P1"], P2_noise=off["P2_noise"], P3_comp=off["P3_comp"], P4_hom=off["P4_hom"],
                Sil=off["Sil"], frag=off["frag"], ARI=off["ARI"], AMI=off["AMI"],
                bg_dom_groups=off["bg_dom_groups"], cand_groups=off["cand_groups"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=",".join(k for k in POOLS if k != "dlrsd"),
                     help="comma-separated pool ids (default: all except bonus dlrsd)")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated source ids")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="feature-extraction device (cuda/cpu). default cuda if available.")
    a = ap.parse_args()
    pool_ids = a.pools.split(",")
    source_ids = a.sources.split(",")

    # _grouping_eval 함수들은 호출 시점에 모듈 전역 DEV 를 참조 — 여기서 override.
    GE.DEV = a.device
    print(f"[setup] pools={pool_ids} sources={source_ids} DIAL={GE.DIAL} DEV={GE.DEV}", flush=True)
    bb = GE.load_backbone(str(REPO / "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"))

    rows = []
    csv_path = OUT / "matrix.csv"
    cols = ["pool", "n", "source", "source_domain", "epoch", "device", "domain_relation", "src_bucket", "tgt_bucket",
            "lf_k", "lf_noise_pct", "lf_non_noise_pct", "lf_over_merge", "lf_stability", "lf_coherence",
            "P1", "P2_noise", "P3_comp", "P4_hom", "Sil", "frag", "ARI", "AMI", "bg_dom_groups", "cand_groups"]
    if csv_path.exists():  # resume: keep prior rows, append new
        with csv_path.open(encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        print(f"[resume] loaded {len(rows)} existing rows from {csv_path}", flush=True)
    done = {(r["pool"], r["source"]) for r in rows}

    def flush():
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in cols})
        (OUT / "matrix.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    for pool_id in pool_ids:
        pool_arg = POOLS[pool_id]
        feats, labels = get_pool_features(pool_id, pool_arg, bb, a.device)
        n = len(labels)
        # frozen baseline — proj=None, once per pool, reused for every source's Δ comparison
        if (pool_id, "frozen_f") not in done:
            z = GE.embedding_from_features(feats, None)
            lf, off = score(z, labels)
            rows.append(row_dict(pool_id, n, "frozen_f", "-", "-", a.device, lf, off)); flush()
            print(f"  {pool_id:16s} frozen_f      k={lf['k']:3d} noise={lf['noise_pct']:5.1f}% P1={off['P1']:>7s} "
                  f"Comp={off['P3_comp']:.3f} Hom={off['P4_hom']:.3f} ARI={off['ARI']:.3f} frag={off['frag']:.2f}", flush=True)

        for src in source_ids:
            if (pool_id, src) in done:
                continue
            ckpt_rel, src_domain = SOURCES[src]
            ckpt = REPO / ckpt_rel
            if not ckpt.exists():
                print(f"  [skip] {src}: checkpoint missing {ckpt}", flush=True)
                continue
            ep = int(re.search(r"ep(\d+)", ckpt.name).group(1))
            head = load_head(ckpt).to(GE.DEV)
            z = GE.embedding_from_features(feats, head)
            lf, off = score(z, labels)
            rows.append(row_dict(pool_id, n, src, src_domain, ep, a.device, lf, off)); flush()
            print(f"  {pool_id:16s} {src:16s} ep{ep:<3d} k={lf['k']:3d} noise={lf['noise_pct']:5.1f}% P1={off['P1']:>7s} "
                  f"Comp={off['P3_comp']:.3f} Hom={off['P4_hom']:.3f} ARI={off['ARI']:.3f} frag={off['frag']:.2f}", flush=True)
    print(f"[OUT] {csv_path}", flush=True)
    print(f"[OUT] {OUT / 'matrix.json'}", flush=True)
    print("[DONE_ZEROSHOT]", flush=True)


if __name__ == "__main__":
    main()
