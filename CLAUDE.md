# CLAUDE.md

이 파일은 Claude Code(claude.ai/code) 새 세션에 프로젝트 진입점을 알려준다.

## 이 repo가 하는 일 (현재)

WM-811K wafer 분포를 학습하고 chip-internal 패턴을 합성해 36 클래스 fail-bit
palette PNG + positions JSON 데이터셋을 생성. Contrastive learning용 합성 데이터.

git에 commit된 파일은 `contrastive.py`, `cnn_yolo.py` 2개뿐이지만 (사용자가 초기로
reset 후 새로 시작) 실제 작업물은 untracked로 다수 존재. **모든 spec과 변경 history는
docs/image-generation/ 와 .claude/ 에 기록**되어 있어 새 세션에서 즉시 이어 작업 가능.

## 새 세션 진입 순서

1. **자동 로드 (이미 들어와 있음)**
   - `~/.claude/projects/D--project-unknown-contrastive/memory/MEMORY.md`
   - 사용 가능 skill 리스트 (system reminder)

2. **프로젝트 컨텍스트** — `MEMORY.md`의 `project_wafer_synthetic_v1.md` 읽기

3. **상세 spec** — `docs/image-generation/`
   - `README.md` (인덱스), `SPEC.md`, `PIPELINE.md`, `CLASSES.md`, `OUTPUT.md`

4. **사용자 피드백 누적** — `.claude/skills/pixel-design/SKILL.md` v1~v13 history 표

5. **현재 진행 상태** — `docs/image-generation/STATUS.md` (background 작업 등)

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `_sample_gen.py` | 메인 generator (multiprocessing). FTN/QTN 자동 포함. |
| `_sample_gen_gpu.py` | GPU 가속 generator (single-proc + ThreadPool) |
| `_fq_metadata.py` | synthetic `partid`/`part_id`/`pgm` + FTN/QTN key/value 생성 |
| `_verify.py` | 데이터셋 검증 (filename/PNG/JSON 스키마/분포 sanity) |
| `_dist_heatmaps/` | WM-811K cca/* 학습된 heatmap 8 클래스 (.npy + .png, repo 포함) |

## 출력 위치

- PNG: `D:/project/data/wm-811k/unknown/<class>/*.png`
- JSON: `D:/project/data/positions/unknown/<class>/*.json`
- positions JSON은 `partid`, `pgm`, `ftn_keys`, `qtn_keys`, chip별 `f`/`q` 포함.
  FTN/QTN hot item은 `b >= 200` defect/invalid chip 분포와 맞춰 크게 생성.
  **목적: FTN/QTN ↔ fail-bit map cross-correlation 분석.** 클래스마다 hot index 셋이
  다르고, defect chip의 hot item 평균은 normal chip 대비 ≥3x (실측 3.5-4.9x).
  새 클래스/spec 변경 시 이 분석성 검증 필수.

## Skills/Agents

데이터 생성:

| Skill | Agent | 용도 |
|---|---|---|
| `image-generation` | `image-generation` | _sample_gen.py 실행 wrapper |
| `image-verification` | `image-verification` | _verify.py 실행 wrapper, read-only |
| `pixel-design` | `pixel-design` | spec 변경 reasoner |
| `gpu-acceleration` | `gpu-acceleration` | GPU 가속 sample 생성 (single-proc + ThreadPool) |

CNN 분류기 (open-set, Normal 제외 학습):

| Skill | Agent | Slash | 용도 |
|---|---|---|---|
| `cnn-plan` | `cnn-plan` | `/cnn-plan` | 학습 plan 수립 (subset YAML, hparam) |
| `cnn-training` | (없음) | `/cnn-train` | `cnn_train.py` wrapper (가드 없이 직접) |
| `cnn-inference` | `cnn-inference` | `/cnn-predict` | `cnn_predict.py` wrapper, threshold sweep |
| `cnn-pipeline` | `cnn-pipeline` | `/cnn-pipeline` | train→Normal predict→threshold 추천 chain |
| `cnn-analyze` | `cnn-analyze` | `/cnn-analyze` | 학습 결과 진단, 다중 run 비교 |

자원 가드 team (RAM 80% / GPU 90% 한계 자동 polling, master/monitor 분리):

| Agent | 역할 |
|---|---|
| `cnn-master` | 학습 dispatch + kill + resume orchestrator (slash `/cnn-train-safe`) |
| `resource-monitor` | 측정·polling·watchdog. abort signal만 master에 반환 |

운영: `/cnn-train-safe <cnn_train.py args>` — team_name=`cnn-team`. master가 monitor 호출해 시작 점검·학습 중 watchdog. 한계 초과 시 process kill + `log/<run>` `_PAUSED` rename(삭제 절대 금지) + 자원 회복 polling + 재시작.

Contrastive (legacy):

| Skill | Agent | 용도 |
|---|---|---|
| `model-training` | `model-training` | `contrastive.py` / `experiments/run_experiment.py` wrapper |
| `evaluation` | `evaluation` | ARI/NMI/purity/silhouette → eval_summary.json |
| `composite-map` | `composite-map` | cluster top-K medoid composite PNG |

**백본 정책 (TAPT)**: contrastive.py의 `LOCAL_BACKBONE_WEIGHTS`는 ImageNet FCMAE pth가 아니라
`cnn_train.py` 결과 `log/<run>/best_model.pth`를 가리킨다. 같은 wafer 데이터로 33-class
supervised 학습된 backbone을 init으로 써서 도메인 정렬된 mid-level feature를 그대로 활용
(sequential transfer / Task-Adaptive Pre-Training). backbone LR은 head 대비 낮게 (e.g.
1e-6 vs 1e-3) 또는 마지막 stage만 unfreeze 권장. detail은
`.claude/skills/model-training/SKILL.md` "백본 초기화 정책".

세부는 각 `.claude/skills/<name>/SKILL.md` 안에서 어떤 docs를 읽어야 하는지 명시.

## CNN classifier (open-set)

데이터: `D:/project/data/wm-811k/unknown/<class>/*.png` (33 defect class). Normal class는
**학습 제외**, inference 시 max_prob threshold로 분류.

Quickstart:

```bash
# 1. smoke test (33 × 20 = 660 샘플, 2 epoch)
python cnn_train.py --epochs 2 --subset-config experiments/quick.yaml --batch 8 --model-tag smoke

# 2. baseline (effective class_weight, EMA on)
python cnn_train.py --epochs 30 --batch 16 --model-tag baseline

# 3. predict + threshold
python cnn_predict.py --model log/<run>/best_model.pth --input <dir> --threshold 0.7 --output preds.json

# 4. threshold sweep (label inferable from folder)
python cnn_predict.py --model log/<run>/best_model.pth --input val_dir --threshold-sweep 0.1,0.9,0.05
```

출력 컨벤션: `log/{model_tag}_{YYYYMMDD_HHMMSS}_F{f1:.2f}_R{recall:.2f}/`
- `best_model.pth` (state_dict + classes + ema_state)
- `hparams.yaml`, `history.json`, `eval_summary.json`
- `val_per_class_report.txt`, `test_per_class_report.txt`
- `best_confusion_matrix_val.png`, `best_confusion_matrix_test.png`
- `curves.png`

도메인-safe augmentation (cnn_train.py::build_transforms):
- ✅ HFlip, ±15° rotation, 작은 translate/scale, Gaussian noise
- ❌ VFlip, 180° rotation (Edge-Top↔Edge-Bottom 손상)
- ❌ ColorJitter (palette grade 의미 손상)

Subset YAML (`experiments/<plan>.yaml`):
```yaml
classes:
  Donut_scratch: 30      # 소수 클래스 시뮬
  Edge-Bottom_scratch: 30
  default: 200
```

## 외부 참조 (read-only, 수정 금지)

- `D:/project/fail-map/` — palette/파일명/JSON 원본 spec
- `D:/project/fail-map/docs/*.md` — 5 문서 (확장자 .md)
- `D:/project/data/wm-811k/cca/<Class>/*.png` — WM-811K 8 클래스 학습 데이터
- `D:/project/data/positions/fq_missing_test/` — JSON 참조 sample
- `D:/project/mapviewer/` — composite map 공식 원본 (다음 stage)

## 절대 금기

- 데이터 폴더 (`D:/project/data/wm-811k/unknown`, `D:/project/data/positions/unknown`)
  무단 삭제 금지. 사용자 명시 요청 전.
- 사용자 피드백 누적된 v1~v13 spec 무근거 변경 금지. 변경 시
  `.claude/skills/pixel-design/SKILL.md` 누적 표 필수 업데이트.
- `transparency=31` PNG save 금지 (모델 입력 픽셀 손실).
- `D:/project/fail-map/`, `D:/project/mapviewer/` 수정 금지.
