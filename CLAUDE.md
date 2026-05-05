# CLAUDE.md

이 파일은 Claude Code(claude.ai/code) 새 세션에 프로젝트 진입점을 알려준다.

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
| `evaluation` | `evaluation` | val 이미지 + 학습된 모델 → ARI/NMI/purity/silhouette → `eval_summary.json`. cosine silhouette. |
| `composite-map` | `composite-map` | cluster top-K medoid composite PNG. 공식 (자매 repo `mapviewer/`) 무수정. |

세부는 `.claude/skills/<name>/SKILL.md`.

## 외부 참조 (read-only)

- `D:/project/known-cnn/` — 자매 repo (supervised CNN, 데이터 합성)
- `D:/project/data/wm-811k/cca/<Class>/*.png` — WM-811K 8 클래스 학습 데이터
- `D:/project/mapviewer/` — composite map 공식 원본

## 절대 금기

- `outputs/logs_contrastive/` 사용자 명시 요청 전 무단 삭제 금지
- `contrastive.py` 직접 수정 금지 — wrapper 의 CFG override 만 사용 (`run_contrastive.py`,
  `_contrastive_n50.py` 패턴)
- `D:/project/known-cnn/`, `D:/project/mapviewer/` 수정 금지
