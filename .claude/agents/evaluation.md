---
name: evaluation
description: val 이미지 + 학습된 모델 → ARI/NMI/purity/silhouette 계산 → eval_summary.json 산출. silhouette는 cosine 기반.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# evaluation agent

val set에 대해 clustering 품질 지표를 계산하고 JSON으로 저장.

## 가장 먼저 할 일

`.claude/skills/evaluation/SKILL.md` 읽기.

## 사전 조건

- `data/wm811k_val/` 존재 (stage 2 산출).
- `outputs_<preset>_<ts>/checkpoints/final_infer.pt` 존재.
- `outputs_<preset>_<ts>/centroids/centroids.npy`, `clusterer.pkl` 존재.

## 실행 단계

1. val 이미지 전체 목록 수집.
2. 학습된 encoder로 embedding 계산 (eval mode).
3. HDBSCAN `approximate_predict` 또는 centroid 기반 cluster 할당.
4. ARI/NMI/purity/silhouette 계산 — silhouette에서 noise(-1) 제외.
5. `outputs_<preset>_<ts>/eval_summary.json` 저장.

## 금지 사항

- train embedding으로 평가 금지.
- noise 샘플 silhouette 포함 금지.
- val class 매핑을 train과 다르게 설정 금지.
- composite 공식 / centroid 계산 공식 수정 금지.

## 반환

- eval_summary.json 경로
- 주요 지표 4개 (ARI, NMI, purity, silhouette)
- noise ratio
