# Dataset

## 1. Source

- **WM-811K** (Wu et al. 2015) — 811 K real wafer maps, 8 distribution classes.
- 우리는 분포만 차용, 실제 image 는 합성으로 대체 (privacy + 일관 control).

## 2. 합성 (자매 repo `known-cnn`)

### 2.1 Wafer
- **6400 × 6400 8-bit palette PNG**.
- 32 × 32 grid of 200 × 200 chips.
- Palette indices:
  - 0-7: defect grade (0=clean, 1-7=defect intensity)
  - 8-23: border / special
  - 24-31: background / invalid

### 2.2 Class taxonomy (현재 baseline)

총 39 GT class:
- 38 **defect class** = 8 distribution × 5 chip-object (일부 조합)
  - distribution: `Center`, `Donut`, `Edge-Bottom`, `Edge-Top`, `Edge-Ring`, `Full`, `Thick-Edge`, plus 9 wafer-canvas
  - chip-object: `bank_boundary`, `fork`, `scratch`, `scratch_rot`, `invalid_main` (round-26 spec)
  - 명명: `<distribution>_<object>` (e.g., `Edge-Top_scratch_rot`)
- 1 **Normal class** = `Normal_bank_boundary` (no defect)

Round 26 명명 변경:
- 이전 `particle_blast` → `fork`
- 이전 `scratch_21deg` → `scratch_rot`
- 사유: 더 일반화된 의미 (각도 21° hardcode 안 함)

### 2.3 Wafer-canvas (9 patterns)
정규 distribution 외 추가:
- `BrokenRing`, `CenterDonut`, `CrescentArc`, `CrossScratch`, `DiagonalSmear`,
- `ParallelScratches`, `RingDots`, `Row`, `Starburst`
- 모두 wafer-level alpha mechanism (`dist_apply/_sample_canvas_gen.py`).

### 2.4 Sample 분포 (baseline `overall`)

총 **8,357 wafer**:
- Normal: 1,000
- 38 defect class: 평균 ~193 / class (50~200+ 범위, Thick-Edge_fork 50)

다음 학습부터 사용자 명시 random 분포 적용 (`docs/contrastive-eval/PRODUCTION.md`):
- Normal 과다 (production 80% 흉내)
- defect 안에서도 imbalanced

## 3. Production 시나리오 (사용자 정정)

```
실제 production (10000 wafer):
  ├─ Normal: ~8000 (80%)
  └─ Defect: ~2000 (20%)
      ├─ defect_a (1차): ~1500
      ├─ defect_b (드물게): ~300
      ├─ defect_c (희귀): ~200
      └─ unknown defect: 가끔 등장
Label 가능: <1% (약 100장)
```

## 4. Label 정책

- 합성 데이터는 합성 시점 GT label 알려져 있음 (folder name = class).
- production 은 대부분 unlabeled. 일부 (~1%) 만 operator review 후 labeled.
- 학습은 GT label 사용 안 함 (SSL — InfoNCE 는 augmentation positive 만 필요).
- 평가는 합성 GT label 사용 (Tier 1+2 metric 산출).

## 5. 발견된 sub-style 변종 (Iter 0)

진단 결과 (`docs/contrastive-eval/DECISIONS.md` D-10):
- `Full_scratch_rot`, `Full_fork`, `Full_bank_boundary`, `Thick-Edge_fork` 4 class 가
  HDBSCAN 으로 항상 두 sub-cluster 로 split.
- 검증: HDBSCAN sweep / intra·inter cosine ratio (2-9×) / GMM bimodality BIC (강한 bimodal).
- **결론**: encoding / clustering 정확. **합성 데이터 자체에 두 sub-style 존재**.
- **처방**: 사용자 결정 대기 — (a) 합성 코드 통일 또는 (b) GT class 두 sub-class 분리.

## 6. 외부 source / read-only

- `D:/project/data/wm-811k/unknown/<class>/*.png` — 합성 wafer (입력)
- `D:/project/data/wm-811k/cca/<Class>/*.png` — WM-811K 원본 학습 (heatmap 학습용)
- `D:/project/known-cnn/dist_apply/` — 합성 코드 (read-only, 우리 repo 외)

## 7. 변경 history

- 2026-05-05 baseline: 8,357 wafer, 39 class (overall run)
- 다음 (Iter 1): production-realistic random class size sampling
