# Contrastive Evaluation — 종합 인덱스

contrastive 학습 / HDBSCAN clustering 의 성능 평가 + 학습 monitoring + production 운영
정책 문서. 사용자 7~8 라운드 합의 사항 통합.

## 한 페이지 요약

### 우선순위 (P1 → P4)
1. **class_capture_rate** — 모든 defect class 가 ≥1 group 에 잡힘 (현재 1.000)
2. **noise_pct (defect only)** — defect 격리 실패 비율 (현재 0.71%)
3. **Completeness** — 같은 class 가 같은 group 에 (현재 0.9466)
4. **Homogeneity** — group 안에 한 class 만 (현재 0.9154)
보조: AMI 0.9288

### 발표/논문 표 1행
```
[Tier 1] Completeness=0.947 AMI=0.929 noise_def=0.71% capture=38/38(1.000)
[Class frag] coverage=0.993 single_cluster=34/38(0.895)
```

### 거부한 옵션
- 커스텀 metric (weighted_isolation, pure_rate, binary_*) — 학술 신뢰성 X
- 분류기 metric (precision/recall/F1/FPR) — clustering 본질 X
- Multi-crop (SwAV) — wafer 위치 정보 손상
- SupCon 주력 — unknown defect generalization 위험
- Tier 3 metric (NMI, V-measure, FMI, DB, CH) — 발표 X (디버그만)

### 채택한 개선
- **Hard Negative Mining** (Robinson 2021) — InfoNCE 위 β param. label 무관, production 호환.
- **alignment + uniformity** (Wang-Isola 2020) — 매 epoch monitoring (label 무관).
- **k-NN top-1** — label 있는 작은 subset 만 옵션.

## 5 문서

| 문서 | 내용 |
|---|---|
| [METRICS.md](METRICS.md) | Tier 1/2/3 공식 metric 정의 + 출처 + 우선순위 + 산출 위치 + 콘솔 보고 형식 |
| [MONITORING.md](MONITORING.md) | 학습 도중 alignment + uniformity (label 무관) + k-NN (label 옵션) + 주기적 HDBSCAN |
| [HARD_NEGATIVE.md](HARD_NEGATIVE.md) | InfoNCE → Hard Mining 수학 (변수 한 줄씩) + β 효과 + production 적용 |
| [PRODUCTION.md](PRODUCTION.md) | 실제 production 시나리오 (Normal 80% / defect 20% / label 1%) + class imbalance 처리 + unknown defect |
| [DECISIONS.md](DECISIONS.md) | 11 결정 history. 거부한 대안 + 채택 사유 (D-1 ~ D-11) |

## 빠른 검색

| 의문 | 답 |
|---|---|
| 이 metric 출력해도 돼? | METRICS.md → Tier 1+2 만 OK |
| 학습 중 collapse 어떻게 봐? | MONITORING.md → uniformity loss 평탄 |
| Hard mining 어떻게 도입? | HARD_NEGATIVE.md → β=1.0 시작 |
| 왜 SupCon 안 해? | DECISIONS.md D-5 |
| 왜 Multi-crop 안 해? | DECISIONS.md D-4 |
| Full_*** split 어떻게? | DECISIONS.md D-10 |

## 시작 — 새 세션 시 추천 진입 순서

1. `README.md` (이 파일)
2. `METRICS.md` (Tier 1+2 만 보고 / 출력)
3. 사용자 질문 따라 해당 sub-document 참조

## 코드 변경 시점

본 docs 는 정책 문서. 실제 코드 변경 (eval pipeline 통합 / hard mining 도입) 은 별도
plan 으로 진행. 코드 변경 시:
- `_eval_contrastive_unknown_n50.py` 통합 — METRICS.md + MONITORING.md 따름
- `contrastive.py` Hard mining — HARD_NEGATIVE.md 따름

## Cross-reference

- skill: `.claude/skills/contrastive-eval/SKILL.md` — 위 docs 의 핵심 압축
- agent: `.claude/agents/evaluation.md` — 본 정책 enforce
- memory: `~/.claude/projects/D--project-unknown-contrastive/memory/` 의 5 entry
- root: `CLAUDE.md` 의 "Contrastive 평가 정책" 섹션
