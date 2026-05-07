# CLAUDE.md

이 파일은 Claude Code(claude.ai/code) 새 세션에 프로젝트 진입점을 알려준다.

---

## ★ 절대 규칙 (Absolute Rules) — 매 turn 끝에 검사

### 규칙 #1 — 모든 turn 종료 시 변경 / 생성 파일 리스트 출력

**언제**: tool 사용으로 파일이 1 개라도 수정 / 생성된 turn 의 마지막
**무엇을**: 다음 형식의 출력 1 회

```
## 이 turn 결과물

### 수정된 파일
- D:\절대경로\file1.md  (수정 — 한 줄 요약)
- D:\절대경로\file2.py  (수정 — 한 줄 요약)

### 새로 생성된 파일
- D:\절대경로\new_dir\file3.md  (신규 — 한 줄 요약)
- D:\절대경로\figs\image1.png    (신규 — 한 줄 요약)
```

**대상 종류**:
- `.md` `.py` `.json` `.yaml` `.txt` 등 source / config 파일
- `.png` `.jpg` `.svg` 등 image
- `parquet` `.npy` 등 데이터 산출물
- 새 디렉토리 생성도 명시

**제외 대상**:
- `outputs_contrastive_*/` 학습 자동 산출물 (run_dir 만 한 줄 언급)
- `_dispatch_logs/` boot log
- 조회 / Read 만 한 파일 (수정 X)
- tool 호출 결과로만 잠시 받은 임시 변수

**왜**: 사용자가 conversation 끝에 변경 사항을 한 눈에 추적 가능 + 빠진 파일 (image, .md 보고서 등) 누락 방지.
**위반**: 파일을 수정 / 생성하고도 마지막에 리스트 안 보여주면 절대 규칙 위반 — 즉시 추가 출력.

---

## 이 repo가 하는 일

**Self-supervised contrastive learning + HDBSCAN unknown wafer-defect clustering.**
WM-811K 분포 + 합성 wafer fail-bit (자매 repo `known-cnn` 의 `dist_apply/` 가 생성)
를 입력으로 contrastive feature 학습 → embedding → HDBSCAN 클러스터링 → cluster
별 medoid composite map 시각화.

자매 repo: [`known-cnn`](https://github.com/hogil/known-cnn) — supervised open-set
CNN 분류기 (33 known + Normal=unknown). 본 repo 는 unsupervised side 만 담당.

## 새 세션 진입 순서

1. **자동 로드** — `~/.claude/projects/D--project-unknown-contrastive/memory/MEMORY.md`
2. **명령어 / 워크플로우** — `USAGE.md`
3. **사용 가능 skill** — system reminder 의 `model-training` / `evaluation` / `composite-map`

## 주요 스크립트 (root, 8개)

| 파일 | 역할 |
|---|---|
| `contrastive.py` | main training engine. CFG block 위에 hyperparameter, backbone path. |
| `contrastive_unknown_n50.py` | unknown 클래스 강조 변종. |
| `run_contrastive.py` | Windows env wrapper. `cnn_train` 결과 backbone state_dict 추출 후 contrastive.py 로 inject. |
| `_contrastive_n50.py` | small-budget 학습 wrapper (per-class 50, normal 200, epochs 20, batch 16). subset hardlink builder 포함. |
| `_contrastive_unknown_n50.py` | _contrastive_n50 의 unknown 변종. |
| `_eval_contrastive_n50.py` | val embedding → ARI/NMI/silhouette/purity. |
| `_eval_contrastive_unknown_n50.py` | 위의 unknown 변종. |
| `predict_contrastive_daily.py` | production daily wafer 폴더 → parquet 출력 (preds + clusters + medoids + review). |

## 데이터 / 출력

- **입력 데이터**: `D:/project/data/wm-811k/unknown/<class>/*.png` (자매 repo 합성)
- **Backbone**: 자매 repo `known-cnn` 의 `outputs/logs_<kind>/overall/best_model.pth` (TAPT —
  같은 wafer 데이터로 supervised pretrain 한 ConvNeXtV2). `run_contrastive.py` 가
  state_dict 만 추출해 contrastive.py CFG 의 `LOCAL_BACKBONE_WEIGHTS` 로 set.
- **출력**: `outputs/logs_contrastive/<tag>_<TS>/`
  - `best_model.pth`, `embeddings.npy`, `metadata.json`
  - `clusters.parquet`, `medoids/<cluster_id>/*.png`, `composite_map_<cluster_id>.png`
  - `eval_summary.json` (ARI / NMI / purity / silhouette cosine)

## Skills / Agents (3 skill, 3 agent)

| Skill | Agent | 용도 |
|---|---|---|
| `model-training` | `model-training` | `contrastive.py` / `_contrastive_n50.py` wrapper. CFG override 정책, 결과 폴더 삭제 금지. |
| `evaluation` | `evaluation` | val 이미지 + 학습된 모델 → 공식 metric (Tier 1+2 + class_fragmentation_summary) → `eval_summary.json`. **커스텀 metric 출력 금지** — `docs/contrastive-eval/` 정책 참조. |
| `composite-map` | `composite-map` | cluster top-K medoid composite PNG. 공식 (자매 repo `mapviewer/`) 무수정. |
| `contrastive-eval` (skill) | (skill only) | eval pipeline 적용 표준 — Tier 1+2 산출 / 콘솔 보고 / 커스텀 metric 금지. agent 가 evaluation 호출 시 자동 활용. |
| `paper-recorder` | `paper-recorder` | ★ 연구 진행 자동 기록 — `docs/paper/` 8 section 누적 update. 매 milestone (학습 완료 / design 변경 / 분석 발견) 마다 invoke. ITERATIONS append-only. |

세부는 `.claude/skills/<name>/SKILL.md`.

### paper-recorder 사용 (★)

연구 진행을 paper-friendly 누적 markdown 으로 기록. invocation:
```
"최신 학습 결과 paper 에 기록"   → ITERATIONS / RESULTS / EXPERIMENTS update
"이 design 변경 반영"             → METHOD / DATASET update
"abstract 갱신"                   → ABSTRACT v 번호 증가 + rewrite
"iteration 시작"                  → ITERATIONS placeholder
```

산출 위치: `docs/paper/{README, ABSTRACT, METHOD, DATASET, EXPERIMENTS, RESULTS, ITERATIONS, REFERENCES, FIGURES}.md` 8 파일.

ITERATIONS.md 는 append-only — 과거 iteration 결과 수정 금지.

## Contrastive 평가 정책 (★)

contrastive 학습 결과 보고 / 평가 시 다음 정책 강제. 자세히는 `docs/contrastive-eval/`.

### 우선순위 (P1 → P4)
1. **class_capture_rate** — 모든 defect class 가 ≥1 group 으로 잡힘 (recall 느낌)
2. **noise_pct (defect only)** — defect 격리 실패 비율 (precision 느낌)
3. **Completeness** — 같은 class 가 같은 group 에
4. **Homogeneity** — group 안에 한 class 만
보조: AMI / Silhouette (cosine) / ARI

### Tier 1 (필수 발표 표 1행, 4 + class_fragmentation_summary)
- **Completeness** (Rosenberg-Hirschberg 2007)
- **AMI** (Vinh et al. 2010)
- **noise_pct (defect only)** (HDBSCAN 표준)
- **class_capture_rate** (`class_fragmentation.parquet` aggregate)

### Tier 2 (보조)
- Homogeneity, Silhouette (cosine), ARI

### 절대 금지 — 커스텀 metric (사용자 명시 거부)
- `weighted_isolation`, `pure_rate`, `mixed_rate`, `isolation`, `contamination_rate`
- `binary_*` (binary_ari / binary_nmi / binary_homogeneity 등 모두)
- precision / recall / F1 / FPR / accuracy / TP/FP/FN/TN — 분류기 style 일체

### Tier 3 — skip (디버그만, 발표 X)
- NMI, V-measure, Fowlkes-Mallows, Davies-Bouldin, Calinski-Harabasz

### 거부한 학습 측 옵션
- **Multi-crop (SwAV)** X — wafer 위치 정보 손상 (D-4)
- **SupCon 주력** X — unknown defect generalization 위험 (D-5)
- USE_LOCAL=True (grid spatial contrast) + Hard Negative Mining (β param) 만 채택

### 학습 monitoring (label 무관)
- 매 epoch: alignment + uniformity (Wang & Isola 2020)
- 옵션 (label 있을 때): k-NN top-1, periodic HDBSCAN

### 다음 학습 dispatch 시
- BATCH=16, IMAGE_SIZE=384 (사용자 명시 GPU 작게)
- **Data anchor (★ 전제)**: defect class 별 평균 30 (random 분포, 일률 X) + Normal 전체
  - 첫 dispatch 시 file_list.parquet 자동 저장 → 재현 보장
  - 이후 모든 method ablation 은 same subset 디렉터리 재사용 (data anchor 변경 금지)

세부 결정 history: `docs/contrastive-eval/DECISIONS.md` D-1 ~ D-15.

## 외부 참조 (read-only)

- `D:/project/known-cnn/` — 자매 repo (supervised CNN, 데이터 합성)
- `D:/project/data/wm-811k/cca/<Class>/*.png` — WM-811K 8 클래스 학습 데이터
- `D:/project/mapviewer/` — composite map 공식 원본

## 절대 금기

- `outputs/logs_contrastive/` 사용자 명시 요청 전 무단 삭제 금지
- `contrastive.py` 직접 수정 금지 — wrapper 의 CFG override 만 사용 (`run_contrastive.py`,
  `_contrastive_n50.py` 패턴)
- `D:/project/known-cnn/`, `D:/project/mapviewer/` 수정 금지
