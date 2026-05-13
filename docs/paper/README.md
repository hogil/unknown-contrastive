# Paper draft — continuous record

본 폴더는 contrastive 학습 + HDBSCAN 클러스터링 wafer defect 연구의 **누적 paper draft**.
`paper-recorder` agent 가 매 milestone 마다 자동 update.

## 8 section

| 파일 | 내용 |
|---|---|
| `ABSTRACT.md` | 1 페이지 요약 (Method + 핵심 결과 + 의의) |
| `METHOD.md` | 데이터 합성 / 모델 아키텍처 / 학습 프로토콜 / 추론 / clustering |
| `DATASET.md` | WM-811K 기반 합성 dataset spec, class 분포, label 정책 |
| `EXPERIMENTS.md` | 실험 setup (GPU, batch, hyperparameters, sampling) |
| `RESULTS.md` | 실험 결과 표 (run 별 Tier 1 metric 누적) |
| `ITERATIONS.md` | 시간순 변경 log (각 iteration 의 변경 + 효과) |
| `REFERENCES.md` | 인용 논문 + 사용 metric 출처 |
| `FIGURES.md` | 보유 figure (composite map, t-SNE 등) + caption |

## 사용 — paper-recorder agent

```
"학습 끝났으니 paper 에 기록"  → agent 가 ITERATIONS + RESULTS + EXPERIMENTS 업데이트
"이 design 변경 반영"           → agent 가 METHOD 또는 DATASET 업데이트
"abstract 갱신"                  → agent 가 ABSTRACT 재생성 (latest results 반영)
```

agent spec: `.claude/agents/paper-recorder.md`
skill: `.claude/skills/paper-recorder/SKILL.md`

## 자동 update source

agent 가 매 invoke 시 다음을 점검:
- `git log --oneline -20` — 최근 commit 메시지
- `outputs/logs_contrastive/<latest>/eval/eval_summary.json` — metric
- `outputs/logs_contrastive/<latest>/run.log` — hyperparameter
- `docs/contrastive-eval/DECISIONS.md` — 채택/거부 history
- 사용자 직접 입력

## 현재 baseline (Iter 0)

`outputs/logs_contrastive/overall/` (= `normal1000_n50_b16_global_e10_resize_reuse_260505_110513`):

| metric | value |
|---|---|
| Completeness | 0.9466 |
| AMI | 0.9288 |
| noise_pct (defect) | 0.71% |
| class_capture_rate | 38/38 = 1.000 |
| frac_single_cluster | 0.8947 (34/38) |
| Silhouette (cosine) | 0.5664 |
| ARI | 0.7002 (over-cluster 페널티 inherent) |
| alignment | 0.3018 (intra-class proxy) |
| uniformity | -2.4955 |

상세: `RESULTS.md`, `ITERATIONS.md` Iter 0.

## 현재 SOTA (★ N1 v7 FINAL — iter 70 NEW, anchor `avg30_new_260508_123037`, ★ revised 2026-05-13)

| metric | value |
|---|---|
| Completeness | 0.983 ± 0.007 |
| AMI | 0.950 ± 0.009 |
| ARI HDBSCAN (3-seed mean) | **0.859 ± 0.018** |
| ARI Agglo Ward K=42 (3-seed mean) | **0.9014 ± 0.022** |
| noise_pct (defect, HDBSCAN) | 1.48% (3-seed mean) |
| class_capture_rate | 43/43 = 1.000 |
| HDBSCAN cfg | eom mcs=12 ms=3, defect-only, no eps |

★ v6 reference numbers (iter 37 / B5 family, **superseded by v7 on multi-seed**):
Completeness 0.991, AMI 0.960, ARI 0.870 single-seed / 0.866 ± 0.014 3-seed, noise 0.61%.
B5 multi-seed reproducibility (RESULTS §17b): 2-seed avg 0.8343 ± 0.031 (HDBSCAN) /
0.8920 ± 0.062 (Agglo K=42) — std 1.7-2.8× higher than NEW; B5 seed=42 0.9358 Agglo claim
retracted as cherry-picked outlier (seed=1 reproduce = 0.8482, Δ −0.088).

iter 50-58 (2026-05-10) 9 추가 iter 의 atomic sweep (LW / LR / NEG / TEMP / TOPK / QUEUE
+ Spatial NeCo Hierarchical / Zone-Aware) 모두 multi-seed std 안 → **iter 37 = multi-axis
saturation point** 확정 (N5 contribution).

상세: `RESULTS.md` 표 7/11/12, `ITERATIONS.md` iter 37 + iter 50-58, `ABSTRACT.md` v0.3.

## ★ Real Baseline Component Isolation (iter 60-65, 2026-05-11)

6-step Real Baseline ablation B0→B5 (Global InfoNCE only → iter 37 cfg) 완료:

| step | cfg | ARI | noise | Comp | AMI |
|:-:|---|---:|---:|---:|---:|
| B0 | Global only | 0.8231 | 6.20% | 0.9602 | 0.9290 |
| B1 | + Local DenseCL | 0.8514 | 3.93% | 0.9665 | 0.9387 |
| B2 | LW=1.0 isolated | 0.8231 | 6.20% | 0.9602 | 0.9290 |
| B3 | + MoCo Queue | 0.8464 | 1.31% | 0.9828 | 0.9496 |
| **B4** ★ | + NEG=0.72 | **0.8605** | **0.52%** | **0.9852** | **0.9557** |
| B5 | + NeCo 0.2 (=iter 37) | 0.8564 | 0.96% | 0.9801 | 0.9503 |

**★ 핵심 발견 N6 (NEW) — Component Interaction Matters**:
- LW lever isolated effect 는 negative (B1→B2 ARI -0.028)
- LW 의 진짜 효과 = Queue interaction (B2→B3 ARI +0.023, noise -4.89pp)
- NeCo (paper N1) isolated effect ≈ 0 (B4→B5 ARI -0.004)
- B4 > B5 (NeCo 없는 cfg 가 모든 metric 우위)
- B5 vs iter 37 (same seed): ΔARI 0.014 = multi-seed std → N2 강화

paper contribution N1-N5 → **N1-N6** 갱신 (Component Interaction NEW).
상세: `RESULTS.md` 표 13, `ABLATION_PLAN.md`, `DISCUSSION.md` §7.9, `ITERATIONS.md` iter 60-65, `ABSTRACT.md` v0.4.

## ★ 2026-05-12 N1 v7 FINAL (iter 84, B5 seed=1 reproducibility retracts v6 absolute SOTA) — ★ CURRENT

iter 84 (`outputs_contrastive_260512_114525/`) 의 B5 seed=1 reproducibility 측정 결과 →
v6 "B5 absolute SOTA Agglo Ward K=42 ARI 0.9358" claim **retracted** (cherry-picked
lucky outlier). B5 seed=1 Agglo K=42 ARI = **0.8482** (Δ −0.0876).

| Method | B5 2-seed avg ± std | NEW 3-seed avg ± std | Δ (NEW − B5) | B5/NEW std ratio |
|---|---:|---:|---:|---:|
| HDBSCAN | 0.8343 ± 0.031 | **0.859 ± 0.018** | **+0.0245** | 1.7× |
| Agglo Ward K=42 | 0.8920 ± 0.062 | **0.9014 ± 0.022** | **+0.0094** | **2.8×** |
| KMeans K=42 | 0.8540 ± 0.044 | **0.8678 ± 0.026** | **+0.0138** | 1.7× |

→ **NEW > B5 on multi-seed avg across all 3 clustering methods** with 1.7-2.8× lower std.
→ dual-cfg recipe → **single-cfg recommendation (NEW)** + dual clustering target.
→ N2 (multi-seed methodology) strongest evidence: B5 seed=42 → seed=1 Δ ARI −0.088 =
   largest cross-seed flip in 84-iter cycle.

상세: ABSTRACT v0.9 (CURRENT), RESULTS §17, DISCUSSION §7.10.7 + §7.12.4 (revised),
CONCLUSION §8.6 + §8.8 (single-cfg), METHOD §3.6 + §3.7, INTRODUCTION C7 (v6→v7),
manager_report SUMMARY §0.7 + REPORT Phase 2 결론, ITERATIONS iter 84 entry.

## 2026-05-12 N1 v6 (complementary at single-seed Agglo K=42) — superseded by v7 on multi-seed

iter 82-83 의 Agglomerative Ward K=42 결과를 per-GT-class purity 로 decompose
한 결과 (RESULTS §16) — paper N1 v5 "NeCo functionally equivalent to Local
DenseCL — substitutable on partitioning" 을 **N1 v6 final: complementary at
per-class scope** 로 refine.

| evidence | B5 (Local + NeCo) | NEW (NeCo only) | Δ |
|---|---:|---:|---:|
| Agglo Ward K=42 ARI single-seed=42 | **★ 0.9358** | 0.9200 | +0.0158 |
| avg per-class purity (Agglo K=42) | **97.0%** | 96.2% | −0.83pp |
| Edge-Ring_fork (n=31) | **100%** | 64.5% | −35.5pp (B5 win) |
| Center_scratch (n=40) | **95%** | 75% | −20pp (B5 win) |
| Donut_fork (n=37) | **100%** | 81.1% | −18.9pp (B5 win) |
| Edge-Top_scratch (n=19) | **100%** | 84.2% | −15.8pp (B5 win) |
| CenterCircle (n=42) | 54.8% | **100%** | +45.2pp (NEW win) |
| Edge-Top_fork (n=20) | 90% | **100%** | +10pp (NEW win) |

→ **dual-cfg recipe RETRACTED in v7** (★ 2026-05-13): see ★ N1 v7 FINAL section above
(line 98). v6 per-class purity observations (RESULTS §16) preserved as single-seed only.
Multi-seed Agglo K=42: NEW 0.9014 ± 0.022 > B5 0.8920 ± 0.062 (RESULTS §17b).

→ **Single-cfg recipe (NEW)** for both frontiers (multi-seed authoritative):
- Frontier 1 (unknown-K + HDBSCAN) → NEW
- Frontier 2 (known-K + Agglo Ward K=42) → NEW (same cfg)

→ v6 framing "Local DenseCL NOT deprecated" superseded by v7: Local DenseCL is
**operationally optional**, not required for SOTA. v5 "substitutable on HDBSCAN
aggregate" preserved. 자세히: ABSTRACT v0.9, RESULTS §17, ITERATIONS iter 84.

## ★ 2026-05-12 Sil retraction (paper N8)

이전 "Sil +30% multi-seed robust" headline 은 HDBSCAN protocol mismatch artefact
(B5 leaf+ms=4, NEW eom+ms=3). apples-to-apples (eom+mcs=12+ms=3, defect-only) 재측정
후 B5 Sil = 0.7988, NEW Sil = 0.7860 → **equivalent within seed variance (−0.013)**.
"+30% Sil" / "Sil +0.184" / "geometry Pareto" 표현 모두 retract.

진짜 NEW vs B5 차이 = ARI marginal (+0.003 3-seed avg) + Normal-cluster consolidation
(paper N1 v5: Normal noise 77.7% → 14.1%, 859/1000 Normals → 1 dense cluster,
full-set ARI 0.83 vs 0.69).

자세한 정정: ABSTRACT v0.6, RESULTS §14c / §14h / §14i / §14k, DISCUSSION §7.10 / §7.11.

## 절대 룰

- 자체 정의 metric (weighted_isolation, pure_rate 등) 절대 출력 금지 — Tier 1+2 공식만
- 분류기 metric (precision/recall/F1/FPR/accuracy) 절대 사용 금지
- 합의 변경 시 `docs/contrastive-eval/DECISIONS.md` 에 D-N 추가 후 paper 반영
- ITERATIONS.md 는 append-only — 과거 iteration 수정 금지
- ★ cross-cfg Silhouette/ARI/noise 비교는 HDBSCAN 모든 axis + metric scope 통일 필수 (paper N8)
