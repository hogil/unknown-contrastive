# CLAUDE.md

> **정본(canonical): `docs/ABSOLUTE_RULES.md`.** 목적·사다리·데이터 경계·에이전트 프로토콜의
> 구속력 있는 기록은 그 파일이다. 아래는 그 요약 + Claude Code 세션용 실무 규칙이며,
> 충돌 시 `docs/ABSOLUTE_RULES.md` 가 우선한다.
> 측정 규율과 에이전트 구조도는 `docs/GOAL_AND_LADDER.md`, 실행 스냅샷은 `docs/HANDOFF_260726.md`.

## ★★★ 절대규칙 0: 이 프로젝트의 목적과 배포 사다리 (사용자 directive, 260726)

> 여러 데이터셋에서 contrastive 성능 향상을 확인하고 ablation 해서 최고 성능을 만들어낸다.
> **최종 목적은 사내 실제 데이터셋에서 unknown grouping 으로 신규불량 발생 시 감지해내는 것.**
> 이걸 위해 끊임없이, 몇 일이 걸려도 진행한다.

**배포 사다리 — 좋은 순서대로 (사용자 명시, 변경 금지)**

| 순위 | 방법 | 검증 상태 (260726) |
|---|---|---|
| **1** | 여기서 만든 **모델로 바로 predict** (zero-shot) | ❌ **실패 확정** — cca 14 source 중 frozen 을 이기는 게 0개 |
| **2** | 여기서 만든 **레시피로 학습** 후 predict | ✅ 작동 — severstal(산업 결함)에서 전 지표 우세 |
| **3** | 여기서 만든 **recipe sweep 학습** 후 predict | ⏳ 진행 — severstal 10셀 승자 `lr008`(LR 2배), ARI +62% |
| **4** | **CNN TAPT** 까지 한 후 학습 sweep 후 predict | 미착수 |

**모든 작업은 이 사다리 중 어느 칸을 전진시키는지 말할 수 있어야 한다.** 말할 수 없으면 우선순위가 낮다.

### 이 목적에서 파생되는 하위 절대규칙

1. **사내엔 라벨이 없다.** "성능이 올랐다"는 주장은 **"라벨 없이 그 설정을 고를 수 있었나"** 까지
   답해야 완결된다. 아니면 오라클 선택이고 배포 불가.
   - epoch: **Rule C** ✅ / 레시피: `argmin(seed_noise)` ⏳ / 다이얼: **bootstrap 안정성** ✅ (260727)
2. **대조군은 frozen 이 아니라 z0(랜덤 head)** 이고 **용량(head 수·차원)을 맞춰야** 한다.
   랜덤 head 만으로도 지표가 잡음폭 밖으로 움직인다.
3. **reassign 전/후를 분리 보고**하라. 후처리 후 noise 는 어떤 임베딩에든 나오는 바닥값이다.
4. **다이얼은 pool 마다 다시 정한다.** 다른 pool 값 이식 금지 (mcs6 이식으로 결론이 뒤집힌 전례).
   ★ **불량 종수 k 는 입력하지 않는다 — 모르는 게 전제고, HDBSCAN 을 쓰는 이유가 k-free 라서다.**
   다이얼은 `min_cluster_size` 의 원래 의미(**"몇 장 이상 뭉쳐야 그룹인가"**, 운영 판단)로 정한다
   — `deploy/config.py::Cluster.MCS` / `MS` 에 **직접** 넣는다.
   (260728: MIN_GROUP_SIZE 유도 + AUTO_DIAL 안정성 스캔 경로는 제거했다. "자동"이 두 뜻으로
    쓰여 혼란만 컸다. 안정성으로 고르고 싶으면 mcs 를 몇 개 돌려 직접 비교하라.)
   실측: 안정성 기준이 severstal 의 아는 정답(mcs20)을 집었다. DBCV 는 15 로 빗나갔고
   ARI 최대화는 mcs60(k=2 병합 치팅)으로 걸어갔다.
5. **측정 지점 하나에서 결론 내지 마라.** 260726 하루에 같은 실수 4건(다이얼·대조군·임계·후처리).
   각 arm 을 **자기 최적점**에서 비교하고, **여러 다이얼에서 순위가 유지되는지** 확인한다.
   ★ ARI 최대화로 다이얼을 고르지 마라 — 전부 한 덩어리로 병합할수록 ARI 가 오른다.

상세: `docs/GOAL_AND_LADDER.md`, `docs/HANDOFF_260726.md`

---

## ★★★ 절대규칙: 데이터셋은 `E:/data/images/` 에만 두고 거기서 로드한다 (260726)

1. **다운로드·생성·로드 위치는 `E:/data/images/<dataset>/` 하나뿐이다.** 새 데이터셋을 받거나
   렌더링하면 반드시 이 경로 아래에 만든다. `D:/project/unknown-contrastive/data/images/` 에
   데이터셋을 두지 않는다.
2. **동일 데이터셋에서 하위셋용 폴더를 새로 만들지 않는다.** subset/split/pool 은 전부
   **manifest(코드)로 선택**한다 — `data/pools/<name>.json` + `scripts/_common.py::resolve_pool()`.
   물리 복사로 파생 폴더를 만드는 행위 금지 (과거 164GB 중복의 원인).
2-1. **★ 하드링크·심볼릭 링크·junction 절대 금지.** 어떤 형태의 링크도 만들지 않는다
   (`os.link`, `mklink`, `fsutil hardlink`, `New-Item -ItemType SymbolicLink` 전부 금지).
   해결책은 링크가 아니라 **원본 코드에서 이미지 선택 경로를 바꾸는 것**이다 —
   master 경로 하나만 두고 코드가 필요한 이미지를 골라 읽는다.
3. manifest 포맷: `{"root": "E:/data/images/<dataset>", "files":[{"path":"<class>/<file>","label":"<class>"}...]}`
   생성기 = `scripts/make_pool_manifest.py`. `--pool` 인자는 디렉토리/manifest 둘 다 받는다(후방호환).
4. 정렬 규칙은 스크립트마다 다르므로 `resolve_pool()` 을 쓸 것 — 임의 정렬 금지
   (임베딩 행 순서 ↔ 라벨 매칭이 깨지면 모든 지표가 조용히 망가진다).

현재 master: `E:/data/images/unknown` (21,143장 / 45 class). 파생 pool 은 `data/pools/*.json` 참조.


이 파일은 Claude Code(claude.ai/code) 새 세션에 프로젝트 진입점을 알려준다.

## 이 repo가 하는 일 (현재)

WM-811K wafer 분포를 학습하고 chip-internal 패턴을 합성해 36 클래스 fail-bit
palette PNG + positions JSON 데이터셋을 생성 → supervised open-set CNN 분류기 학습.
33 defect class 학습, Normal 은 inference 시 max_prob threshold 로 unknown 처리.

자매 repo: `unknown-contrastive` (contrastive learning + HDBSCAN unknown clustering).
known-cnn 은 supervised side 만 담당.

## 새 세션 진입 순서

1. **자동 로드 (이미 들어와 있음)**
   - `~/.claude/projects/D--project-known-cnn/memory/MEMORY.md`
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
| `_sample_gen.py` | obj-active 18-class generator (multiprocessing). FTN/QTN + chip-object crop 자동 포함. round 25: invalid 비례 fix (defect.sum() * 0.15). |
| `_sample_gen_gpu.py` | GPU 가속 generator (single-proc + ThreadPool). chip-object crop 동일 포함. |
| `_sample_canvas_gen.py` | **9 obj-less wafer-canvas generator** (round 12-25). chip-internal alpha 매커니즘 wafer 6400×6400 한 번에 적용. Lorentzian sharp + heavy tail sum. alpha = baseline ↔ peak mix. chip border = alpha mean primary filter. 자세한 spec: `docs/image-generation/CANVAS_9.md`. |
| `_fq_metadata.py` | synthetic `partid`/`part_id`/`pgm` + FTN/QTN key/value 생성 |
| `cnn_train.py` / `cnn_predict.py` | 통합 engine. 직접 호출 가능, 또는 kind 별 wrapper 가 호출. |
| `cnn_train_chip.py` / `cnn_train_wafer.py` | engine 의 thin wrapper. config 맨 위에. |
| `cnn_train_compound.py` | wafer 33-class 3-channel (R=failbit, G=obj_id, B=zero) compound 학습. logs_compound/. |
| `cnn_predict_chip.py` / `cnn_predict_wafer.py` / `cnn_predict_compound.py` | kind 별 추론 entry. overall/ 자동 + run dir 자동. |
| `cnn_eval_chipgrid.py` | **chip-grid 32×32 native** wafer 분류 평가 (standalone, 기존 trainer 무수정). obj_id encoding 변종 sweep (V0~V6) + chip CNN noise robustness. logs_chipgrid/. 자세한 건 `docs/chipgrid/`. |
| `chip_tools/_build_obj_id_maps.py` | chip CNN 으로 wafer 마다 obj_id 맵 inference 생성. incremental save. |
| `chip_tools/_chip_resample_100.py` | classification_chips/ 를 N/class 만 keep + archive (top-N by defect_ratio). |
| `chip_tools/_chip_trim_inplace.py` | classification_chips/ in-place trim (archive 없이 그냥 N/class 만 남기고 삭제). |
| `chipgrid_eval/_chipgrid_gmm_options.py` / `_chipgrid_summary.py` | chipgrid eval 분석 sub-tools. |
| ~~`dist_learn/`~~ | ★ 이 repo 에서 제거(260727). 정본은 자매 repo `known-cnn/dist_learn/`. 여기엔 .py 없이 생성물만 복사돼 있었다. 필요하면 known-cnn 에서 `python dist_learn/_dist_learn.py` 로 재생성하고 `HEATMAP_DIR` 로 넘겨라. deploy/ 파이프라인은 안 쓴다. |
| `verify_tools/_verify.py` | 데이터셋 검증 (filename/PNG/JSON 스키마/분포 sanity) |
| `misc/cnn_yolo.py`, `misc/download_backbone.py` | legacy + utility. |
| `_dist_heatmaps/` | WM-811K cca/* 학습된 heatmap 8 클래스 (.npy + .png, **gitignored**). 부재 시 `python dist_learn/_dist_learn.py` 로 재생성. |

## 출력 위치

- PNG: `D:/project/data/wm-811k/unknown/<class>/*.png`
- JSON: `D:/project/data/positions/unknown/<class>/*.json`
- positions JSON은 `partid`, `pgm`, `ftn_keys`, `qtn_keys`, chip별 `f`/`q` 포함.
  FTN/QTN hot item은 `b >= 200` defect/invalid chip 분포와 맞춰 크게 생성.
  **목적: FTN/QTN ↔ fail-bit map cross-correlation 분석.** 클래스마다 hot index 셋이
  다르고, defect chip의 hot item 평균은 normal chip 대비 ≥3x (실측 3.5-4.9x).
  새 클래스/spec 변경 시 이 분석성 검증 필수.
- **chip-object crop**: `D:/project/data/wm-811k/classification_chips/<obj>/<wafer_basename_without_yield_sys>_x<x>_y<y>_b<bin>.png`
  - 5 obj label (bank_boundary / fork / scratch / scratch_rot / invalid_main)
  - **wafer generation 시점에 inline 저장** (`_sample_gen.save_chip_crops`).
    chip별 true object 라벨(`chip_meta['obj']`)을 사용하므로 75% primary + 25% mixed
    환경에서도 정확. **post-process folder-suffix 라벨링 절대 금지** (25% mixed chip 오라벨).
  - source positions JSON 에 `chips[].obj` 추가 금지 (정책)

## Skills/Agents

데이터 생성:

| Skill | Agent | 용도 |
|---|---|---|
| `image-generation` | `image-generation` | _sample_gen.py + _sample_canvas_gen.py 실행 wrapper. obj-active 18 + canvas 9 |
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
| `wafer-classifier` | (없음) | — | ★ V3 chipgrid (1.16M, val_f1 0.9946) + 비교 base (R-only, obj-only, compound) 학습/평가/추론. 표 정책 + active class YAML + block_expand + TTA 금지 정책 enforced. docs/wafer-ensemble/ 인덱스. |
| `multi-label-ablation` | `multi-label-ablation` | — | 8-stage paper-style multi-label ablation orchestrator. ★ loss / matching / 판정 의 mix 조합 sweep |
| `chipgrid-eval` | `chipgrid-eval` | — | **chip-grid 32×32 native** wafer 분류 평가 (`cnn_eval_chipgrid.py`). obj_id encoding 변종 sweep (V0~V6) + chip CNN noise robustness. 작은 데이터·빠른 ablation. V3 (one-hot 5ch) 가 데이터 4× 적게 + 모델 76× 작게 compound 동등 도달. docs/chipgrid/. |

Multi-label ablation (★ 진행 중):

| 위치 | 역할 |
|---|---|
| `docs/multi-label/README.md` | 전체 인덱스 (8 stage + 3 deep-dive) |
| `docs/multi-label/THEORY.md` | multi-label / SPML / calibration / density 이론 + 수식 |
| `docs/multi-label/LOSS_DESIGN.md` | ★ deep-dive: loss 단일 + mix 조합 7 (M1-M7) |
| `docs/multi-label/MATCHING_DESIGN.md` | ★ deep-dive: chip-wafer matching ensemble + CRF (C1-C7) |
| `docs/multi-label/DECISION_RULE.md` | ★ deep-dive: multi-label 판정 (D1-D8) |
| `docs/multi-label/PAPERS.md` | 인용 논문 list + 우리 도메인 적용 |
| `docs/multi-label/EXAMPLES.md` | benchmark 사례 (MixedWM38 / COCO / OpenImages SPML) + 실측 수치 |
| `docs/multi-label/STAGES.md` | 8 stage motivation + H1-H8 가설 + 기대 효과 |
| `docs/multi-label/STATUS.md` | 진행 상태 + 산출 path |
| `~/.claude/plans/1-input-batch-hidden-patterson.md` | 8-stage 실행 plan (실행 detail) |
| `.claude/skills/multi-label-ablation/SKILL.md` | 실행 패턴 + sweep range + 최적값 찾는 방법 |
| `.claude/agents/multi-label-ablation.md` | stage orchestrator agent spec |
| `_dist_learn_per_class.py` | ✅ Stage 1 완료 (5 method × 33 class × 5 data-amount) |
| `D:/project/data/wm-811k/unknown_multi/` (예정) | Stage 3 합성 데이터 (multi-label GT) |
| `_dist_heatmaps_per_class/` (gitignored) | Stage 1 산출 surface 850 npy |
| `results/`, `plots/` (gitignored) | 모든 stage 산출 CSV + plot |

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

**백본 정책**: `models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth` (ImageNet FCMAE
pretrained ConvNeXtV2-base, 88M). 모든 33-class wafer / 5-class chip / compound trainer
의 init backbone. local-only — `cnn_train.py` 가 자동 로드.

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
- ❌ HFlip: scratch_rot 등 angle = 클래스 정체성 (21° → -21°)
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

## ★★★ 절대 규칙: 이미지 생성/표시 후 출력 폴더 전체 경로를 메시지 마지막에 출력 필수 ★★★

이미지 (chip / wafer / preview / sample / figure 등 어떤 PNG/JPG/이미지 산출물이든)
**생성 직후, 그리고 그 이미지 / 폴더를 언급/표시할 때마다, 사용자에게 출력 폴더의
절대 경로 전체를 메시지 마지막 줄에 무조건 표시**.

❌ 금지:
- 이미지 생성하고 결과만 보여주고 경로 안 알려주기
- 사용자가 "어디에 만들었지?" 물을 때까지 기다리기
- 상대 경로 `_pink_preview/foo.png` 만 보여주기
- **이미 만들었던 이미지를 또 보여줄 때 경로 생략하기** (사용자 가장 자주 분노 원인)
- **메시지 중간에만 경로 적고 마지막에 안 적기** — 마지막 줄에 무조건 다시
- 분석/요약/결정 question 등으로 메시지 끝나면서 경로 안 적기

✅ 의무 — **모든 이미지 관련 메시지의 맨 마지막 줄**:
- `[OUT] D:/project/known-cnn/_chip_revert_preview/`  (절대 경로, drive letter 포함)
- 여러 폴더면 list 로 모두 표시
- 새 생성 / 기존 재표시 / 비교 / 분석 모두 동일 — 매번 매 메시지 마지막에 적기
- 메시지 마지막에 결정 질문 있어도 그 아래에 [OUT] 한 줄 더 추가

배경 명령도 동일 — `[OUT]` 줄을 stdout 에 찍고 응답에도 포함.

이 규칙 위반 = 사용자가 매번 "어디 만들었어?" 다시 묻게 됨 → 시간 낭비 + 사용자 분노.
특히 같은 폴더 여러 메시지 걸쳐 다룰 때 매번 마지막에 다시 적기.

## 절대 금기

- 데이터 폴더 (`D:/project/data/wm-811k/unknown`, `D:/project/data/positions/unknown`)
  무단 삭제 금지. 사용자 명시 요청 전.
- 사용자 피드백 누적된 v1~v13 spec 무근거 변경 금지. 변경 시
  `.claude/skills/pixel-design/SKILL.md` 누적 표 필수 업데이트.
- `transparency=31` PNG save 금지 (모델 입력 픽셀 손실).
- `D:/project/fail-map/`, `D:/project/mapviewer/` 수정 금지.

## Active class policy (33 → 20 active + 14 archive → 27 with canvas 9)

V3 chipgrid (val_f1 0.9946) 의 saturated 분류 결과 기반으로 **33 wafer class →
20 active + 14 archive** 결정. 학습 시 active list 만 사용, 데이터는 보존.

이후 round 12-25 wafer-canvas 9 추가 → **active 27** (`experiments/active_classes_27.yaml`):
- 18 obj-active (Donut/Edge-Bottom/Edge-Top × 5 obj + Edge-Ring × 2 + Thick-Edge_invalid_main)
- 9 wafer-canvas (DiagonalSmear, CrossScratch, CrescentArc, ParallelScratches, BrokenRing,
  RingDots, CenterDonut, Row, Starburst). spec: `docs/image-generation/CANVAS_9.md`.
- 제외: `Center_invalid_main`, `Full_invalid_main` (V3 chipgrid saturated, archive 보존)

원본:
- **Active 20** (`experiments/active_classes_20.yaml`): Donut×5, Edge-Bottom×5,
  Edge-Ring×4 (-invalid_main), Edge-Top×5, Thick-Edge_invalid_main
- **Archive 14** (`experiments/archive_classes_14.yaml`): Center×5, Full×5,
  Edge-Ring_invalid_main, Normal_bank_boundary, Starburst, CommaCluster
- 데이터 보존: archive 14 class 는 `D:/project/data/wm-811k/unknown_archive/<class>/`,
  `D:/project/data/positions/unknown_archive/<class>/` 로 **copy** (원본 삭제 X)
- obj_id_maps 는 영향 X (flat basename lookup)
- Canvas 9 의 Row 만 직접 PIL Draw line (사용자 round 24 명시), 나머지 8 은 alpha-based

지원 trainer: `cnn_eval_chipgrid.py`, `cnn_train_compound.py`, `cnn_train_objonly.py`,
`_chipgrid_kde_gmm.py`, `cnn_train_chipgrid_fusion.py` 모두 `--active-classes-yaml` 지원.
strict 기본 — `--allow-missing-active-classes` 없으면 YAML class 가 data dir 에 없으면 fail.

선정 기준 / 상세: `docs/wafer-ensemble/ACTIVE_CLASSES.md`,
`~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`.

❌ 금지:
- `unknown/<class>` 데이터 폴더 무단 삭제 (글로벌 룰 + 본 정책)
- `EXCLUDE_CLASSES` 같은 hardcoded list 에 새 class 추가 (active YAML 사용)
- archive 폴더 (Starburst, CommaCluster 등) 삭제 — 8 wafer-canvas 새 class 와 같이 활용 예정

## Block expand policy (categorical resize — BICUBIC/NEAREST hardcode 금지)

obj_id (32×32 categorical) / one-hot binary / probability 등 **categorical map 의
spatial resize 는 `_chipgrid_resize.block_expand_2d` 만 사용**. PIL/torch 의
BICUBIC, NEAREST, F.interpolate(...) 모두 코드 hardcode 금지.

```python
# ✅ 올바른 사용
from _chipgrid_resize import block_expand_2d
obj_384 = block_expand_2d(obj_32, 384, 384)            # 정수 12 px/cell
obj_200 = block_expand_2d(obj_32, 200, 200)            # 6 px/cell + 8 cell 7px (균등 spread)

# ❌ 금지 — categorical 신호 깨짐
PIL.Image.fromarray(obj_32).resize((384, 384), Image.BICUBIC)
F.interpolate(obj_t, size=(384, 384), mode='nearest')   # 정수 배수 가정
```

이 정책이 V3 chipgrid (val_f1 0.9946) 의 enabling factor:
- compound R+G+B 384 BICUBIC ceiling 0.9784 (val) — obj_id 정수 BICUBIC 보간이 신호 망가뜨림
- V3 32×32 native + one-hot 5ch = 보간 0 → val_f1 0.9946 (errors 75% 감소)

상세: `~/.claude/projects/D--project-known-cnn/memory/feedback_block_expand_only.md`,
`docs/wafer-ensemble/DISCOVERY.md` D3.

새 trainer 작성 / 기존 trainer 수정 시 BICUBIC/NEAREST 발견하면 사용자에게
즉시 보고 + block_expand_2d 로 패치.

## 표 정책 (wafer 분류 결과 보고)

새 학습 / 비교 결과 보고 시 **반드시** 다음 컬럼 포함 (`.claude/skills/wafer-classifier/SKILL.md`
의 표 정책 + `docs/wafer-ensemble/RESULTS.md` 형식):

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |

- `n train` = 학습 데이터 sample 수
- `epoch (best/total)` = best epoch / total trained (early stop 포함, e.g. "6 / 13 (es)")
- `val_p`, `val_r` = val precision, recall (macro)
- `val_err` = val error count (P != Y 합)
- `params` = total trainable parameters (M 단위)

## ★ Fair-eval protocol (모든 backbone 비교 시 강제)

**한 표 안 row 들은 동일 protocol** — split/active class/epoch/sample 다르면 별 표 분리.

| 항목 | 값 |
|---|---|
| active class (immediate) | `experiments/active_classes_22.yaml` (22 class — 8 obj-less 미합성) |
| active class (target) | `experiments/active_classes_30.yaml` = `configs/chipgrid_class30_target.yaml` (30 class, 8 합성 후) |
| per-class sample | 200 (모든 class 통일, sorted file pick) |
| split | 0.8 / 0.1 / 0.1 stratified, seed 42 |
| split (active 22) | n_train 3520 / val 440 / test 440 |
| split (active 30) | n_train 4800 / val 600 / test 600 |
| epoch | 30 (early stop 끔), best val_f1 epoch model |
| optimizer | AdamW wd 0.05, cosine warmup 3ep |
| batch/lr | small (≤2M) 64/1e-3, large (>10M) 16 / head 1e-3 backbone 1e-4 |
| augmentation | rotate ±15°, translate/scale ±3°, gaussian σ=0.01 (no flip/colorjitter/mixup/cutmix) |
| TTA | 절대 금지 |

spec yaml: `experiments/fair_eval_protocol.yaml`. 설명: `docs/wafer-ensemble/FAIR_EVAL_PROTOCOL.md`.

다른 protocol 결과는 별 표 분리 (header 에 protocol 명시).

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

## chip-multilabel module (260506 unknown-contrastive 에서 이관)

`chip_multilabel/` 의 single-label train → multi-label predict 평가 파이프라인. classification_chips/ 4 class (bank_boundary, fork, scratch, scratch_rot) 학습 + 합성 12-class eval set (4 single + 6 combo + Normal + Invalid) 평가.

### 핵심 파일
- `chip_multilabel/gen_eval_set.py` — eval set 합성 (min-blend combo + BASELINE Normal + orange-border Invalid). `--source-strength-pct N` 로 strong-defect 만 source 사용.
- `chip_multilabel/_train_chip_variant.py` — 8 train variant (T1 CE+LS, T3 Focal, T4 ASL, T5 BCE, T6 BCE→ASL, T7 BCE+LS, T8 CE-soft+CutMix). 9+ CLI hparam flags (LS, ASL γ, EMA, warmup, drop_path, two-LR, CutMix, ...).
- `chip_multilabel/run_stage1.py` — 기존 모델 + 11 inference variants (I0~I10, I5 영구 금지). I7/I10 winner.
- `chip_multilabel/run_phase_a.py` — coordinate-descent hparam sweep (LS/LR/epochs).
- `chip_multilabel/run_stage2.py` — 7 train × 6 inference matrix.
- `chip_multilabel/notes.md` — 실시간 작업 노트 (iter 1~10 + restore point).

### 자율 loop 실험 결과 (iter 1~10, 약 70+ trains)
- **Iter 8 winner (single model)**: T9 (BCE+LS in [0.05,0.10] + CutMix p=0.5 + rect=0.5) — 3-seed mean macro_f1 **0.9305 ± 0.046**.
- **★ Iter 10 final (260506) — H Ensemble winner**: baseline T9d + C_44 (Normal trained, cutmix=0.25) **logit avg** → **10-defect macro F1 = 0.9950**, 5-sample-seed mean **0.9930 ± 0.005**, FAR 0.00% (Normal 80% real-env). 4-single 0.9963, 6-combo 0.9908. 모든 Normal/Invalid F1 1.000 lock.
- **단일 master 폴더 정책**: defect 200 store (`--source-strength-pct 50` 강한 defect) + Normal 200 + Invalid 50 = 2450 chip. runtime `--n-per-class 50` 으로 sample. subset 폴더 절대 안 만듦.
- **Normal training 필수** (4-class only 학습은 Normal F1 huge variance ±0.466). y=-1 sentinel + multi-hot zero-vector target. 사용자 directive "Normal 학습에 들어갔어야" 입증.
- **Logit ensemble = best 약점 보완** (paper finding) — diversity (with-Normal vs without) > quantity (multi-seed). 0.91 → 0.995.
- **Cross-class suppression** — Normal training 이 fork combo prob 0.46→0.16 (3× collapse). ensemble 로 fix.
- **bb+sr recall fix** (iter 7): 0.32 → 0.85+ via CutMix mechanism (compositional 학습).
- **음성 결과** (paper-worthy): warmup, EMA(0.95), drop_path 0.05, cutmix-rect 0.25, two-LR, CE-soft+CutMix, ASL light, F (fork-pair bias retrain) — net negative ensemble.
- **Iter 11 in progress** (paper-style 4-row ablation matrix): 6 loss × 6 inference × 3 eval (p50 simple / p30 simple / p50 diverse Normal) = 108 cells.

### Skills / Agents
- `chip-multilabel-pipeline` (skill) — datagen → stage1 → stage2 풀체인 entry
- `chip-multilabel-runner` (agent) — GPU dispatcher, resource-monitor 협조
- `chip-multilabel-analyst` (agent, opus) — 결과 분석 + 다음 실험 제안 (paper 인용 + 도메인 reasoning)
- `chip-multilabel-logger` (agent) — docs/chip-multilabel/ 기록
- `chip-multilabel-paper-narrator` (agent, opus) — paper section narrative

### 절대 금기 (chip-multilabel)
- TTA 영구 금지 (scratch vs scratch_rot 회전 구분 손상)
- Rotation/Flip aug 영구 금지 (학습 시 RandomAffine translate+scale 만)
- 1 atomic method/iter 변경

자세한 결과는 `docs/chip-multilabel/` 참조.
