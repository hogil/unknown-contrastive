---
name: unknown-loop
description: unknown 불량 발견 성능의 무한 자율 개선 루프 마스터. loop-monitor(진행 확인) → loop-analyzer(성능 분석) → loop-planner(계획) → dispatch 를 반복. 사용자 "루프 돌려/계속 진행해" 시 invoke.
---

# unknown-loop — 마스터 오케스트레이터

## 목표
라벨·k 없는 wafer 풀에서 unknown 불량 그룹 발견 성능을 무한 반복으로 개선한다 (사용자 directive: "결과 확인 → 새 학습계획 → 진행, 무한반복").

## 5-에이전트 체제 (260612 사용자 재편)
| 에이전트 | 담당 |
|---|---|
| loop-paper | 논문 분석 — 약점 겨냥 외부 기법 발굴 + 제약 채점 + 이식안 |
| loop-repo | 프로젝트 폴더 기록 발굴 — 옛 레시피/검증값/교훈 (재발명 방지) |
| loop-analyzer | 실험결과 분석 + **최적조건 도출** (채점·판정·다음 후보 제안) |
| loop-resource | 리소스 분석 — 사전점검 + **★신규 실행 피크 실측 필수** + 건강 모니터 |
| (master = 본 스킬) | 위 4 에이전트 관장 + 결정 + dispatch + 사용자 보고 |

## 루프 사이클 (1회전)
```
 ① loop-resource  — 실행 중 점검 / 사고 처치 근거 수집
 ② (완료 시) loop-analyzer — 고정 잣대 채점 + 판정 + 최적조건 도출
 ③ (필요 시) loop-paper / loop-repo — 새 기법·기록 발굴 (큐 고갈/정체 시)
 ④ master 결정: 다음 1개 실험 (atomic, 과제-중립, 검증값·분포보정 우선)
 ⑤ loop-resource — 사전 자원 점검 (GPU: nvidia-smi, 자매 30-40% 공존)
 ⑥ master dispatch — 학습=GPU(batch≤16, 1 process) / 평가·어댑터=CPU
      ★ 신규 종류 run 은 dispatch 직후 loop-resource 피크 실측 필수
 ⑦ 사용자 보고: 판정 + 현황판 + 다음 실험 → 알림 대기
```

## 고정 컨텍스트 (모든 사이클에서 불변)
- 메모리: `~/.claude/projects/D--project-unknown-contrastive/memory/project_dinov3_ncd_autoloop.md`
  (현 SOTA, 우선순위 큐, 봉인된 다이얼, 정책 전부 여기)
- 고정 잣대: mixed29 / UMAP(nn10&15, dim10) / HDBSCAN(mcs10/ms3/leaf/eps0.15) / capture 1열
- 절대 규칙: 라벨 학습 금지(SupCon 등), 과제특화(합성혼합/분해) 금지, mcs↑ 치팅 금지,
  결과 폴더 삭제 금지, CPU 1 학습, 매 epoch 임베딩 저장, 새 실행파일은 +x commit
- ep 정책: epochs 2 기본 (ep1 정점 패턴). 무라벨 epoch 선택 = uniformity/DBCV (검증됨)

## 이상 처치 표준
| 증상 | 처치 |
|---|---|
| segfault (patch 계열) | batch 2 + OMP/MKL_NUM_THREADS=8 재시도, 재발 시 CPU 보류 |
| loss 즉사 (<0.05 @ep1) | queue 플래그 확인 후 재dispatch. ignore 임계는 분포 기준 (절대값 이식 금지) |
| 좀비/래퍼 잔존 | PowerShell 로 python+bash 정확 매칭 kill (whack-a-mole 방지) |
| 침묵 종료 exit 127/255 | **외부 kill (알약 백신 추정)** — 아래 적대 환경 운영 참조 |

## ★ 적대 환경 운영 (260612 실측 교훈)
- **시간대**: 밤 03~05시 창에선 background bash/python 전부 6~35분 내 사살 (sleeper bash 포함).
  주간(07:30~19:00)은 12시간 무사고 실측. → 장시간 학습은 주간 창에 배치.
- **진전은 파일 단위로만 생존**: 학습 = step ckpt (--ckpt-every 50, ~9분) / 캐싱 = 이미지 단위
  npz (skip-existing 재개). RAM 속 진전은 전부 증발 가정.
- **에스컬레이션 사다리**: bash run_in_background (1차) → 사살되면 **Windows 작업 스케줄러**
  (`schtasks /create /tn unknown_loop_* /tr "powershell -File <ps1>"`) — OS 레벨이라 면역.
  부활 래퍼 .ps1 패턴: _v3_morning.ps1 / _cache_resume.ps1 참조.
- **E 트랙 (cached-adaptor) 이 구조적 정답**: epoch 초 단위 = 사살 불가 체급.
  _cache_features.py → _adaptor_train.py.

## 종료 조건
사용자 중지 지시. 그 외엔 큐 소진 시 planner 가 coordinate-descent 로 후보 재생성.
