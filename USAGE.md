# Usage Guide — contrastive learning + unknown clustering

전체 파이프라인 사용법.

```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│ 0. backbone 준비        │    │ 1. contrastive 학습       │    │ 2. embedding + cluster   │
│ (sister repo known-cnn) │ →  │ run_contrastive.py        │ →  │ HDBSCAN → centroids       │
│ cnn_train_*.py 결과      │    │ _contrastive_n50.py       │    │ medoid composite per      │
│ best_model.pth          │    │ contrastive_unknown_n50.py│    │ cluster                   │
└─────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                          ↓ outputs/logs_contrastive/<tag>_<TS>/
                                          best_model.pth
                                          embeddings.npy + metadata.json
                                          clusters.parquet
                                          medoids/<cid>/*.png
                                          composite_map_<cid>.png
                                          eval_summary.json
```

---

## 1. 학습 (Windows / smoke 기본)

```bash
# default smoke (epochs=2)
python run_contrastive.py

# 본 학습
EPOCHS=20 BATCH=32 python run_contrastive.py
BACKBONE_CKPT=D:/project/known-cnn/outputs/logs_wafer/overall/best_model.pth python run_contrastive.py
```

`run_contrastive.py` 가 자매 repo `known-cnn` 의 `cnn_train` 산출물에서 state_dict 만
추출해 임시 `.pth` 로 저장 후 `contrastive.py` 의 `LOCAL_BACKBONE_WEIGHTS` 로 inject.
contrastive.py 자체 수정 X — env var / CFG override 만.

## 2. small-budget 학습 (per-class 50, normal 200)

```bash
python _contrastive_n50.py \
  --epochs 20 --batch 16 --per-class 50 --normal 200 \
  --backbone D:/project/known-cnn/outputs/logs_wafer/overall/best_model.pth
```

subset hardlink builder 자동 포함 → 작은 학습용 mini-dataset 만들고 contrastive.py 호출.

## 3. unknown 변종

```bash
python _contrastive_unknown_n50.py \
  --epochs 20 --batch 16 --per-class 50 --normal 200
```

unknown class 강조한 sampling.

## 4. 평가 (ARI / NMI / silhouette / purity)

```bash
python _eval_contrastive_n50.py \
  --run outputs/logs_contrastive/<tag>_<TS>/

python _eval_contrastive_unknown_n50.py \
  --run outputs/logs_contrastive/<tag>_<TS>/
```

`eval_summary.json` 산출. silhouette는 cosine 기반.

## 5. Production daily inference

```bash
python predict_contrastive_daily.py \
  --image-root <image_root> \
  --output-root <output_root> \
  --model outputs/logs_contrastive/<tag>_<TS>/best_model.pth \
  --device <product> --processid <line> --date <YYYYMMDD>
```

입력: `<image_root>/<device>/<processid>/<yyyymmdd>/*.png`
출력: `<output_root>/<device>/<processid>/<yyyymmdd>/`
- `manifest.json`, `run.log`
- `preds.parquet`, `clusters.parquet`, `cluster_members.parquet`
- `medoids/<cid>/*.png`, `review/`, `embeddings/`

---

## 6. 외부 dependency (자매 repo)

backbone, 합성 데이터는 자매 repo `known-cnn` 가 담당:

```bash
# 합성 데이터 생성 (known-cnn)
cd D:/project/known-cnn
python dist_apply/_sample_gen.py
python dist_apply/_sample_canvas_gen.py

# CNN supervised 학습 → backbone 산출 (known-cnn)
python wafer_train/cnn_train_wafer.py
# → outputs/logs_wafer/<run>/best_model.pth
```

이 산출 backbone 을 본 repo 가 `BACKBONE_CKPT` 로 받는다.

## 7. 절대 금기

- `outputs/logs_contrastive/` 사용자 명시 요청 전 무단 삭제 금지
- `contrastive.py` 직접 수정 금지 — wrapper 의 CFG override 만 사용
