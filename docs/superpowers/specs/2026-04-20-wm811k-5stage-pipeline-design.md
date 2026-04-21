# WM-811K 기반 5-Stage Skill/Agent 파이프라인 설계

**Date**: 2026-04-20
**Status**: Approved (design phase)
**Author**: hogil (brainstorm with Claude)

## 목표

WM-811K 원본 데이터셋을 기반으로 4000×4000 palette PNG를 생성하고, contrastive
learning → clustering → composite 시각화까지 이어지는 end-to-end 파이프라인을
5개의 독립 skill+agent 쌍으로 모듈화한다.

각 stage는
1. **skill** (rule book; 입출력 계약, 공식, 금기사항, 검증 기준)
2. **agent** (실행자; skill을 invoke한 뒤 도구로 수행)

의 2-파일 쌍으로 구성하며, stage 간 통신은 파일시스템 artifact + manifest
JSON을 통한다. 각 agent는 독립 실행 가능하고, 순서대로 체이닝도 가능하다.

## 파이프라인 개요

```
LSWMD.pkl
    |
    v
[Stage 1] pixel-design    --> configs/pixel_rules.yaml + analysis report
    |
    v
[Stage 2] image-generation --> data/wm811k_train/, data/wm811k_val/
    |
    v
[Stage 3] model-training   --> outputs_<preset>_<ts>/
    |
    v
[Stage 4] evaluation       --> outputs_*/eval_summary.json + plots
    |
    v
[Stage 5] composite-map    --> outputs_*/cluster_summary/composite/*.png
```

## 파일 레이아웃

```
.claude/
├── skills/
│   ├── pixel-design/SKILL.md
│   ├── image-generation/SKILL.md
│   ├── model-training/SKILL.md
│   ├── evaluation/SKILL.md
│   └── composite-map/SKILL.md
└── agents/
    ├── pixel-design.md
    ├── image-generation.md
    ├── model-training.md
    ├── evaluation.md
    └── composite-map.md

configs/
└── pixel_rules.yaml      # stage 1 산출, stage 2 소비

data_prep/
└── wm811k_to_palette.py  # stage 2 구현 (재작성)

data/
├── wm811k_train/<class>/*.png  # stage 2 산출
└── wm811k_val/<class>/*.png    # stage 2 산출

docs/superpowers/specs/
└── 2026-04-20-wm811k-5stage-pipeline-design.md  # 본 문서
```

## Stage 1: pixel-design

**목적**: WM-811K 데이터셋을 실제로 들여다보고, **분석 결과에 근거해** 다음을
**설계 결정**한다:

1. **Upscaling 방법** — 원본 가변 해상도(~26×26~~53×58) → 4000×4000. NEAREST
   vs 그 외 후보. 원본 wafer die 가로/세로 비율, palette 무결성, defect 블록
   크기가 어떤 시각적 결과를 낳는지 분석.
2. **정상 die → palette grade 매핑** — 기본 `1 → 0`이지만, 근거 있어야 함
   (fail-map 공식 palette와의 호환, Grade 0이 정상 die 대표하는 의미).
3. **불량 die → palette grade 분포** — `2 → {1..7}` 매핑의 **비율 설계**.
   WM-811K 원본에는 grade 1-7 세분화 정보가 없으므로 합성적 결정이 필요.
   실제 반도체 데이터 특성(grade 7은 희귀, grade 1-2는 일반적)을 분석·인용한
   근거 기반 weight 도출.

이 단계는 "YAML을 미리 써두는" 설정 단계가 아니라 **데이터를 분석해 규칙을
도출하는 reasoning 단계**다.

**입력**
- `data_raw/LSWMD.pkl`

**수행 작업 (agent가 따라야 할 순서)**

1. **데이터 통계 수집**
   - class별 wafer 수
   - wafer 크기(H, W) 분포 (min/max/median)
   - class별 defect-die 비율 (각 wafer에서 `(wm==2).sum() / (wm>0).sum()`)
   - class별 defect 공간 분포(중앙/엣지/산발) — 간략 지표라도 OK

2. **Upscaling 분석**
   - NEAREST 선택 이유: palette index는 discrete라 interpolation 하면 gradient
     중간값(palette에 없는 index)이 생겨 무효. BILINEAR/BICUBIC은 금지.
   - 원본 크기 대비 upscale factor가 class별로 다름 (~75배~150배). 이로 인한
     defect 블록 시각화 크기 차이를 감안.
   - 대안 검토(예: 원본 해상도 유지 + 가변 캔버스)를 간단히 트레이드오프 설명.

3. **Grade 분포 설계**
   - 정상 die → 0: fail-map 팔레트 계약과 일치.
   - 바깥 → 31: fail-map transparency 규약.
   - defect → 1..7 weight: 후보 설계 2-3개 제시(exponential decay, power-law,
     linear) + 선택 근거.
   - 기본 선택: `1/2^k` exponential decay. 근거 — 실제 wafer에서 grade 7급
     다결함 클러스터는 희귀(~1% 미만), grade 1-2급 isolated/약결함이 대부분.
   - 최종 weight 값은 재현성을 위해 소수점 고정.

4. **Train/Val 풀 계산**
   - 사용자 요구: class당 defect 20(train)+100(val) + normal 4840(train)+1200(val).
   - class별 pool > 20+100 확인. 부족 class 감지 시 경고.

**산출물**
- `configs/pixel_rules.yaml` — stage 2가 소비. 스키마(값은 stage 1 분석 결과):
  ```yaml
  version: 1
  seed: 42
  size: [4000, 4000]
  upscale: nearest
  mapping:
    outside: 31
    normal: 0
    defect:
      mode: random_skewed
      grades: [1, 2, 3, 4, 5, 6, 7]
      weights: [...]       # stage 1이 결정
  split:
    train:
      defect_n_per_class: 20
      normal_n: 4840
      total: 5000
    val:
      defect_n_per_class: 100
      normal_n: 1200
      total: 2000
  ```
- `configs/wm811k_analysis.md` — 위 수행 작업 4단계의 결과를 문서화
  (표 + 간단 설명 + 선택 근거 + 대안과의 트레이드오프).

**규칙 (skill)**
- Upscaling 후보 중 **palette index를 깨뜨리지 않는 방법만** 채택 (NEAREST).
- Weight 후보는 반드시 **정량 근거와 함께** 제시. 감 잡고 찍는 것 금지.
- train/val wafer source **disjoint**, 복원추출 금지.
- 분석 수치는 `configs/wm811k_analysis.md`에서 재계산 가능하도록 재현성 확보.

**검증 기준**
- pool check: 각 class에서 N_train + N_val ≤ 사용 가능 wafer 수.
- `weights`의 합을 1로 재정규화 시 확률 합 1.0 ± 1e-6.
- YAML 스키마 유효성 (version 필드, 필수 키 존재).

## Stage 2: image-generation

**목적**: pixel_rules.yaml + LSWMD.pkl → train/val palette PNG 집합 생성.

**입력**
- `data_raw/LSWMD.pkl`
- `configs/pixel_rules.yaml` (stage 1 산출)

**산출물**
- `data/wm811k_train/<class>/*.png` — class당 정의된 수
- `data/wm811k_val/<class>/*.png` — class당 정의된 수
- `data/wm811k_train/summary.json`, `data/wm811k_val/summary.json` — class별 실제 저장 수

**규칙 (skill)**
- 원본 해상도에서 grade 배정 → NEAREST `Image.resize((4000, 4000))`로 확대.
  (upscale 전 grade를 결정함으로써 defect 블록이 균일하게 확대됨.)
- palette PNG (`mode='P'`), `transparency=31`, `optimize=True`.
- fail-map 파일명 규약: `{CLASS}_00P_W{idx:03d}_{YYYYMMDD}_{HHMMSS}.png`.
- 기존 `assign_grades_by_neighbor_density` / `wm_to_palette` 제거.
- 기존 `data/wm811k/` 존재 시 **사용자 승인 후에만** 삭제 (글로벌 규칙 존중).

**코드 변경**
- `data_prep/wm811k_to_palette.py`:
  - 신규 함수 `assign_grades_random_skewed(wm, rng, grades, weights)` — defect(=2)
    를 weight 기반 random sample.
  - CLI: `--config configs/pixel_rules.yaml`, `--pkl data_raw/LSWMD.pkl`,
    `--train-out data/wm811k_train`, `--val-out data/wm811k_val`.
  - train/val disjoint sampler: class별 `np.random.permutation(n_have)` 후 앞/뒤
    슬라이스.
- 구 함수와 alias 제거.

**검증 기준** (tests)
- normal(wm=1) → palette 0 결정론적.
- outside(wm=0) → palette 31 결정론적.
- defect(wm=2) → {1..7} 범위, 대량 샘플 시 weight 분포 재현 (±2% 허용).
- seed 고정 재현성.
- train/val index 겹침 없음.
- NEAREST resize 후 index 값 보존 (새 grade 생성 안 됨).

**예상 용량**
- 파일당 4-6MB × 7000장 → 약 30-42 GB.

## Stage 3: model-training

**목적**: train set + preset → contrastive 모델 학습, checkpoint + centroid 저장.

**입력**
- `data/wm811k_train/<class>/*.png`
- `experiments/presets.py`의 preset 이름 (기본 `baseline`)

**산출물**
- `outputs_<preset>_<RUN_TS>/` (기존 contrastive.py 구조)
  - `checkpoints/final_infer.pt`, `last_training.pt`
  - `centroids/centroids.npy`, `centroids_meta.json`, `clusterer.pkl`
  - `clusters/hdbscan/cluster_XXX_size_YYY/`
  - `cluster_summary/`, `ignored_samples/`
  - `run.log`, `run_info.json`

**규칙 (skill)**
- 기존 `experiments/run_experiment.py` 인터페이스 유지. 새 CLI 없음, wrapper agent만.
- `INPUT_DIR` = `data/wm811k_train`으로 고정 override.
- 학습 완료 후 `outputs_*/` 절대 삭제 금지 (글로벌 규칙).
- 재실행 시 새 timestamp 폴더 — skip 로직 우회 위한 삭제 금지.

**Agent 동작**
1. stage 2의 `summary.json` 존재 여부 확인 (pre-requisite).
2. `experiments/run_experiment.py --preset <name>` 호출.
3. 생성된 `outputs_<preset>_<ts>/` 경로를 반환 (다음 stage 전달용).

**검증 기준**
- `final_infer.pt` 존재, `centroids.npy` 존재, HDBSCAN `clusterer.pkl` 존재.

## Stage 4: evaluation

**목적**: val set + trained model → cluster 품질 지표 계산.

**입력**
- `data/wm811k_val/<class>/*.png`
- `outputs_<preset>_<ts>/` (stage 3 산출)

**산출물**
- `outputs_<preset>_<ts>/eval_summary.json` — 필드:
  - `ari`, `nmi`, `cluster_purity` (기존 eval_metrics.py 기반)
  - `silhouette` (신규; val embedding의 cosine silhouette)
  - `per_class_assignment` — val class → 할당된 cluster 분포
- `outputs_<preset>_<ts>/plots/` (선택) — confusion 매트릭스, silhouette plot.

**규칙 (skill)**
- val 이미지 → trained encoder → embedding → centroid 기반 cluster 할당.
- silhouette는 cosine distance 기반, val set 크기가 크면 subsample(최대 2000).
- cluster id = -1(noise) 샘플은 metric 계산 제외 + 별도 보고.

**코드 변경**
- `experiments/eval_metrics.py`: silhouette 계산 함수 추가.
- 또는 신규 `experiments/eval_val.py`를 만들어 val 전용 평가 진입점 분리.

## Stage 5: composite-map

**목적**: cluster별 대표 composite PNG 생성 (기존 `cluster_composite.py` 래핑).

**입력**
- `outputs_<preset>_<ts>/clusters/hdbscan/`
- `outputs_<preset>_<ts>/centroids/centroids.npy`

**산출물**
- `outputs_<preset>_<ts>/cluster_summary/composite/cluster_XXX_composite.png`

**규칙 (skill)**
- 기존 `common/composite.py` 공식 유지 (mapviewer `api/composite_map.py`와 호환).
- top-10 medoid 기반 composite. 이 공식 절대 수정 금지.
- 이미 `contrastive.py::main()` 끝에서 자동 호출되지만, 독립 agent로도 호출 가능.

**Agent 동작**
1. `outputs_<preset>_<ts>/` 경로 인자로 받음.
2. `cluster_composite.py` 또는 내부 함수 직접 호출.
3. 생성된 composite 경로 목록 반환.

## 구현 범위 (이 세션)

전체를 한 번에 다 만들기보다 다음 순서로 점진 구축:

1. **5개 skill 파일 전부 작성** — rule book이 먼저 있어야 agent가 따름.
2. **5개 agent 파일 전부 작성** — 스캐폴딩. stages 3-5는 기존 코드 호출 래퍼만.
3. **Stage 1 구현** — `data_prep/analyze_wm811k.py` (신규) + `configs/pixel_rules.yaml`
   초안 생성.
4. **Stage 2 구현** — `data_prep/wm811k_to_palette.py` 재작성 + 테스트 업데이트.
5. **Stage 3-5** — 기존 코드 그대로 래핑 (새 로직 없음). silhouette 계산만 추가.

## 글로벌 규칙 준수

- 결과 폴더 삭제 금지: `outputs_*/`, `data/wm811k_train/`, `data/wm811k_val/` 생성
  후 사용자 명시 요청 전 삭제·덮어쓰기 금지.
- `data/wm811k/` (기존 폴더) 삭제는 사용자가 이미 승인함. 실행 직전 한 번 더 확인 후
  진행.
- 서버 경로(Linux) 변경 금지 — `contrastive.py`의 CFG는 건드리지 않음.
- `fail-map`, `mapviewer` 수정 금지 — read-only 참조.

## 미해결 이슈 (후속)

- `__MULTICROP__` preset은 아직 `NotImplementedError`. stage 3 agent가 해당 preset
  호출 시 명확히 fail하도록 가드.
- Stage 4 silhouette 계산 시 val set 크기(2000) 기준으로는 subsample 없이 직접
  계산 가능. 추후 더 큰 val 셋에서 성능 이슈 시 재검토.
