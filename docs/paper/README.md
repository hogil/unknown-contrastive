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

## 현재 SOTA (iter 37, new anchor `avg30_new_260508_123037`)

| metric | value |
|---|---|
| Completeness | 0.991 |
| AMI | 0.960 |
| ARI (single-seed best) | 0.870 |
| ARI (3-seed mean) | 0.866 ± 0.014 |
| noise_pct (defect) | 0.61% |
| class_capture_rate | 43/43 = 1.000 |
| HDBSCAN cfg | eom mcs=12 ms=3 |

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

## 절대 룰

- 자체 정의 metric (weighted_isolation, pure_rate 등) 절대 출력 금지 — Tier 1+2 공식만
- 분류기 metric (precision/recall/F1/FPR/accuracy) 절대 사용 금지
- 합의 변경 시 `docs/contrastive-eval/DECISIONS.md` 에 D-N 추가 후 paper 반영
- ITERATIONS.md 는 append-only — 과거 iteration 수정 금지
