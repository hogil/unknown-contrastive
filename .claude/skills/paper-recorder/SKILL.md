---
name: paper-recorder
description: 연구 진행 자동 기록 skill — docs/paper/ 8 section 의 update 패턴 + Tier 1+2 공식 metric only + ITERATIONS append-only 정책. invocation 시 source 점검 → 적절 section update.
---

# paper-recorder skill

contrastive 학습 + clustering 의 모든 milestone 을 docs/paper/ 8 section 에 자동 기록.

## 8 section update 패턴

### `ABSTRACT.md` (rewrite OK)
- 250 단어 목표
- Method (1-2 sentence) + 핵심 결과 (Tier 1 numeric) + 의의
- v 번호 (v0.1, v0.2 ...) — milestone 마다 증가
- 갱신 trigger: major iteration 완료, 새 발견

### `METHOD.md` (rewrite OK)
- 섹션 구조: Data synthesis / Backbone / Contrastive training / Clustering / Inference / Composite / Evaluation / Monitoring / 거부 옵션 / 향후
- Design 변경 시 해당 section 업데이트
- 새 paper 인용 시 REFERENCES.md 도 함께

### `DATASET.md` (rewrite OK)
- WM-811K source / 합성 spec / class taxonomy / sample 분포 / production 시나리오
- sampling 정책 변경 시 (e.g., random class size) 업데이트
- 새 class 추가 시 taxonomy table 갱신

### `EXPERIMENTS.md` (table row append)
- Hyperparameter table — run 별 row 추가 (절대 row 삭제 X)
- Hardware / software 변경 시 update
- 이전 run 의 hyperparameter 보존

### `RESULTS.md` (★ table row append)
- 표 1 (Tier 1 metric) 에 새 run row 추가
- 표 2 (Tier 2 보조), 표 3 (class_fragmentation_summary), 표 4 (split classes), 표 5 (alignment+uniformity), 표 6 (retrieval) 에도 동일하게 append
- 과거 run row 절대 수정 X
- 해석 텍스트는 latest 1개만 — 이전 해석은 ITERATIONS 로 이동

### `ITERATIONS.md` (★ APPEND-ONLY)
새 Iter N 추가 형식:
```markdown
## Iter N — <변경 한줄> (YYYY-MM-DD)

### 설정
- (변경된 hyperparameter 만 명시. 변경 없으면 "Iter N-1 동일")

### 결과
- **Completeness=X.XXXX, AMI=X.XXXX, noise_pct=X.XX%, capture=N/N**
- frac_single_cluster=X.XXXX (N/N split)
- alignment=X.XXXX, uniformity=-X.XXXX (Iter 2+)

### 발견
1. (1-3 항목, 짧게)

### 다음 (Iter N+1)
- (계획)
```

### `REFERENCES.md` (entry add)
- 새 method 도입 시 해당 카테고리에 entry 추가
- arxiv ID + 1줄 설명
- 카테고리: Contrastive learning / Clustering metrics / Clustering algorithm / Backbone / Domain / tooling

### `FIGURES.md` (rewrite per figure)
- 새 figure 산출 시 entry update (위치 + caption)
- 보유 figure list + TBD list

## Invocation source 점검 순서

1. `git log --oneline -20 D:/project/unknown-contrastive` — 최근 commit (단서)
2. `ls outputs/logs_contrastive/` — 새 run 있나
3. 가장 최근 run 의 `eval/eval_summary.json` — Tier 1 metric
4. `eval/align_uniform.json` (있으면) — Wang-Isola
5. `run.log` — hyperparameter (CFG dump 또는 grep)
6. `docs/contrastive-eval/DECISIONS.md` — 새 D-N 추가됐는지

## Tier 1+2 표시 형식 (고정)

표 row:
```
| Iter X | YYYY-MM-DD | 0.XXXX | 0.XXXX | X.XX% | N/N (X.XXX) | N/N (X.XXX) |
```
컬럼: Run, Date, Completeness, AMI, noise_pct (def), class_capture, frac_single_cluster.

콘솔 보고:
```
[Tier 1] Completeness=0.947 AMI=0.929 noise_def=0.71% capture=38/38(1.0000)
[Class frag] coverage=0.993 single_cluster=34/38(0.895) mean_n_clusters=1.10
```

## 절대 금지 (사용자 정책 enforce)

- 커스텀 metric (weighted_isolation, pure_rate, binary_*) 사용 X
- 분류기 metric (precision/recall/F1/FPR) 사용 X
- ITERATIONS 과거 entry 수정 X
- 무단 commit X (사용자 명시 시에만)
- Tier 3 metric (NMI/V-measure/FMI/DB/CH) 1차 표 사용 X — 디버그 부록만

## 자동 invocation 시점 (제안)

매 milestone 시 사용자 명시:
- 새 학습 완료 직후
- DECISIONS.md 새 D-N 추가 후
- abstract / paper 정리 시점 (월간 rhythm)

자동 invoke X (사용자 trigger 만).

## 산출 형식

응답 마지막 1-2 줄 summary:
```
[paper-recorder] Iter N appended → ITERATIONS.md
                 RESULTS.md table row: Completeness=X.XXXX AMI=X.XXXX noise=X.XX% capture=N/N
                 files: ITERATIONS.md, RESULTS.md, EXPERIMENTS.md, ABSTRACT.md
```
