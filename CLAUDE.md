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

3. **명령어 / 워크플로우** — `USAGE.md` (image-gen → train → predict 명령어 + 옵션 + 출력 구조)

4. **상세 spec** — `docs/image-generation/`
   - `README.md` (인덱스), `SPEC.md`, `PIPELINE.md`, `CLASSES.md`, `OUTPUT.md`

5. **사용자 피드백 누적** — `.claude/skills/pixel-design/SKILL.md` v1~v13 history 표

6. **현재 진행 상태** — `docs/image-generation/STATUS.md` (background 작업 등)

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `_sample_gen.py` | 메인 generator (multiprocessing). FTN/QTN + chip-object crop 자동 포함. |
| `_sample_gen_gpu.py` | GPU 가속 generator (single-proc + ThreadPool). chip-object crop 동일 포함. |
| `_fq_metadata.py` | synthetic `partid`/`part_id`/`pgm` + FTN/QTN key/value 생성 |
| `cnn_train.py` / `cnn_predict.py` | 통합 engine. 직접 호출 가능, 또는 kind 별 wrapper 가 호출. |
| `cnn_train_chip.py` / `cnn_train_wafer.py` | engine 의 thin wrapper. config 맨 위에. |
| `cnn_train_compound.py` | wafer 33-class 3-channel (R=failbit, G=obj_id, B=zero) compound 학습. logs_compound/. |
| `cnn_predict_chip.py` / `cnn_predict_wafer.py` / `cnn_predict_compound.py` | kind 별 추론 entry. overall/ 자동 + run dir 자동. |
| `_build_obj_id_maps.py` | chip CNN 으로 wafer 마다 obj_id 맵 inference 생성. incremental save. |
| `_verify.py` | 데이터셋 검증 (filename/PNG/JSON 스키마/분포 sanity) |
| `_dist_learn.py` | WM-811K cca/* heatmap 학습 (1회). gitignored heatmap 부재 시 재실행. |
| `_dist_heatmaps/` | WM-811K cca/* 학습된 heatmap 8 클래스 (.npy + .png, **gitignored**). 부재 시 `python _dist_learn.py` 로 재생성. |

## 출력 위치

- PNG: `D:/project/data/wm-811k/unknown/<class>/*.png`
- JSON: `D:/project/data/positions/unknown/<class>/*.json`
- positions JSON은 `partid`, `pgm`, `ftn_keys`, `qtn_keys`, chip별 `f`/`q` 포함.
  FTN/QTN hot item은 `b >= 200` defect/invalid chip 분포와 맞춰 크게 생성.
  **목적: FTN/QTN ↔ fail-bit map cross-correlation 분석.** 클래스마다 hot index 셋이
  다르고, defect chip의 hot item 평균은 normal chip 대비 ≥3x (실측 3.5-4.9x).
  새 클래스/spec 변경 시 이 분석성 검증 필수.
- **chip-object crop**: `D:/project/data/wm-811k/classification_chips/<obj>/<wafer_basename_without_yield_sys>_x<x>_y<y>_b<bin>.png`
  - 5 obj label (bank_boundary / particle_blast / scratch / scratch_21deg / invalid_main)
  - **wafer generation 시점에 inline 저장** (`_sample_gen.save_chip_crops`).
    chip별 true object 라벨(`chip_meta['obj']`)을 사용하므로 75% primary + 25% mixed
    환경에서도 정확. **post-process folder-suffix 라벨링 절대 금지** (25% mixed chip 오라벨).
  - source positions JSON 에 `chips[].obj` 추가 금지 (정책)

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
| `chip-object-dataset` | `chip-object-dataset` | — | inline chip-object crop dataset 검증 (`_sample_gen.save_chip_crops`) |
| `stage3-compound` | `stage3-compound` | `/stage3-compound` | 3-stage 학습 orchestrator: chip 5-class (logs_chip/) → obj_id_map cache → 3-channel compound CNN (logs_compound/) |

학습 / 추론 entry script — **kind 별로 분기된 9개** (학습 3 + dev predict 3 + prod predict 3):

**Dev (단일 입력 폴더, JSON record + wide CSV):**

| Kind | 학습 entry | dev 추론 entry | data root | log root | predict root |
|---|---|---|---|---|---|
| chip 5-class (object) | `cnn_train_chip.py` | `cnn_predict_chip.py` | `data/wm-811k/classification_chips/` | `logs_chip/` | `logs_predict_chip/<TS>_<input>/` |
| wafer 33-class R-only | `cnn_train_wafer.py` | `cnn_predict_wafer.py` | `data/wm-811k/unknown/` | `logs_wafer/` | `logs_predict_wafer/<TS>_<input>/` |
| wafer 33-class compound (R+G) | `cnn_train_compound.py` | `cnn_predict_compound.py` | `data/wm-811k/unknown/` + `obj_id_maps/` | `logs_compound/` | `logs_predict_compound/<TS>_<input>/` |

**Prod (`<image_root>/<product>/<line>/<date>/` 트리 walk, parquet 출력):**

| Kind | prod 추론 entry | 입력 트리 | 결과 트리 (DB ingestion) | row 단위 |
|---|---|---|---|---|
| wafer | `cnn_predict_wafer_prod.py` | `<image_root>/AB/K1AB/<YYYYMMDD>/*.png` | `result_wafer/AB/K1AB/<YYYYMMDD>/preds.parquet` | 1 row / wafer |
| chip | `cnn_predict_chip_prod.py` | 동일 + sibling `<positions_root>/AB/K1AB/<YYYYMMDD>/*.json` (chip rect inline crop) | `result_chip/.../preds.parquet` | 1 row / chip |
| compound | `cnn_predict_compound_prod.py` | 동일 + chip CNN inference inline → obj_id_map → 3ch wafer 추론 | `result_compound/.../preds.parquet` | 1 row / chip (wafer_class 반복, chip_object_class per chip) |

각 prod batch 마다 별도 `logs_predict_<kind>/<TS>_<product>_<line>_<date>/_meta.json` 작성 (model 경로, n_input, n_processed, started/finished_at, status). DB 트리 (`result_*`) 와 분리.

prod 모델 resolve: `--model-glob "logs_<kind>/{line}/overall/best_model.pth"` default — `{line}` 가 batch 의 line dir 명 (e.g. K1AB) 으로 substitute. 글로벌 모델 사용 시 `--model-glob "logs_<kind>/overall/best_model.pth"` 로 override.

`cnn_train_chip.py` / `cnn_train_wafer.py` / `cnn_predict_chip.py` / `cnn_predict_wafer.py` 는 **`cnn_train.py` / `cnn_predict.py` engine 의 thin wrapper** — config 만 맨 위에 박혀있음. engine 직접 호출도 backward compat.

**predict 자동 동작:**
- `--model` 생략 시 `logs_<kind>/overall/best_model.pth` 자동 로드
- 시작 시 `_overall_meta.json` 의 best_run / val_f1 / seeded_at stderr 출력
- `logs_predict_<kind>/<TS>_<input_name>/` 폴더 자동 생성 → `preds.json`, `per_class_report.txt`, `wrong/<true>/<pred>/*.png` 자동 배치 (`--no-run-dir` 로 끄기 가능)

각 logs_*/ 안에 `overall/` 폴더 — 학습 종료 시 val F1 이 그 폴더 내 best 면 현재 run 폴더 통째 복사 교체. `_overall_meta.json` 에 source_run + val_f1 기록.

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

Quickstart (kind별 entry script — config 는 각 .py 파일 맨 위 `# === CONFIG ===` 섹션):

```bash
# 학습 — 각 wrapper 가 default data_dir + log_root 박아둠
python cnn_train_chip.py     --epochs 30 --batch 16 --model-tag chip5
python cnn_train_wafer.py    --epochs 30 --batch 16 --model-tag wafer33
python cnn_train_compound.py --epochs 30 --batch 16 --model-tag compound33

# smoke / subset
python cnn_train_wafer.py --epochs 2 --subset-config experiments/quick.yaml --batch 8 --model-tag smoke

# 추론 — --model 생략 시 logs_<kind>/overall/best_model.pth 자동 (+ _overall_meta.json 출력)
python cnn_predict_chip.py     --input <chip_folder>
python cnn_predict_wafer.py    --input <wafer_folder> --threshold 0.7
python cnn_predict_compound.py --input <wafer_folder> --threshold-sweep 0.1,0.9,0.05

# 명시적 모델 override 도 작동
python cnn_predict_wafer.py --model logs_wafer/<특정_run>/best_model.pth --input <dir>

# 추론 출력 폴더 자동 (logs_predict_<kind>/<TS>_<input_name>/) — 끄려면 --no-run-dir
```

학습 출력 컨벤션: `logs_<kind>/{model_tag}_{YYMMDD_HHMMSS}_{test_f1:.2f}_{val_f1:.2f}/` (3-way),
또는 `logs_<kind>/{model_tag}_{YYMMDD_HHMMSS}_{val_f1:.2f}/` (`--train-val-only`).
default `model_tag` = backbone short name (`convnextv2_base`).

산출:
- `best_model.pth` (state_dict + classes + ema_state + test/val metrics)
- `best_history.txt` — 통합 결과 (4 sections: BEST OVERALL, FINAL per-class, BEST UPDATES SUMMARY, PER-EPOCH PER-CLASS)
- `best_confusion_matrix.png` — combined (test 위 + val 아래, 셀 숫자 annotation)
- `curves.png` (매 epoch 갱신)
- `history.json` (매 epoch 갱신)
- `hparams.{yaml,txt}`, `run.log`
- `wrong/{val,test}/<true>/<pred>/*.png`

폐지: `eval_summary.json`, `val_per_class_report.txt`, `test_per_class_report.txt`,
`best_confusion_matrix_{val,test}.png` — 모두 `best_history.txt` + 통합 PNG에 흡수.

도메인-safe augmentation (cnn_train.py / cnn_train_compound.py::build_transforms):
- ✅ ±15° rotation: 검사장비 stage 회전 오차 범위 내
- ✅ 작은 translate/scale (±3%): alignment / magnification variability
- ✅ Gaussian noise σ=0.01: sensor pixel noise
- ❌ HFlip: scratch_21deg 등 angle = 클래스 정체성 (21° → -21°)
- ❌ VFlip / 180° rotation: Edge-Top ↔ Edge-Bottom 클래스 뒤집힘
- ❌ ColorJitter: palette grade(0=정상, 1-7=강도) 의미 손상
- ❌ MixUp / CutMix / Cutout: palette pixel 평균이 무의미한 grade 생성

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

## 절대 규칙: 클래스/그리드 크기 의존 상수 hardcode 금지

코드 전반에서 **클래스 수 / 그리드 크기 / 카테고리 개수에 의존하는 상수는 hardcode 금지**.
런타임에 source-of-truth (checkpoint, dataset meta, JSON coord 필드, .npy shape) 에서 derive.

이유: 클래스/오브젝트가 늘어날 때마다 코드 여러 곳 손대야 하고, 빠뜨리면 silent bug.

❌ 금지 예시:
- 색 팔레트 `PALETTE_RGB = np.array([... 8 fixed colors ...])`
- 격자 `GRID = 32`, `OBJ_ID_GRID = 32`
- ID 매핑 `OBJECT_TYPE_ID = {"none": 0, "scratch": 2, ...}`
- 서브폴더 가정 `obj_id_maps/<wafer_class>/<basename>.npy` (구조 바뀌면 깨짐)

✅ 올바른 패턴:
- `palette = make_palette(n_chip_objects)` — HSV 균등 분할 N 색
- grid = JSON `coord.tiles_w_rot / tiles_h_rot` 에서 읽기, fallback `max(x_abs)+1`
- ID 매핑 = ImageFolder classes 알파벳 + 1 offset (idx 0 = none)
- npy lookup = flat basename → npy_path map (rglob) — 어떤 서브폴더든 호환

새 코드 / 기존 코드 수정 시 hardcode 흔적 발견되면 사용자에게 즉시 보고하고 동시에 동적 derive 로 패치.
