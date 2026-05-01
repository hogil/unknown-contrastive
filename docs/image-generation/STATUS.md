# Current Status (live)

이 파일은 현재 진행 중인 작업 상태를 기록. 세션 reboot 후에도 새 세션이 이걸 읽고
이어 작업 가능. **변경 시 timestamp 갱신**.

## 마지막 업데이트
2026-05-01

## 진행 중 작업

### Background generation: 7200 samples (200 × 36 classes)

- 시작: 2026-05-01 (대략)
- 명령: `python _sample_gen.py --n 200 --workers 8`
- 예상 소요: 3-4시간
- PNG 출력: `D:/project/data/wm-811k/unknown/<class>/*.png`
- JSON 출력: `D:/project/data/positions/unknown/<class>/*.json`
- 임시 로그: `/tmp/gen_200.log` (이전 세션의 tee 출력, 새 세션에선 없음)

### 진행 확인 방법

```bash
# 현재까지 생성된 PNG 수
find D:/project/data/wm-811k/unknown -name "*.png" | wc -l
# 예상 최종: 7200 (36 × 200)

# 클래스별 진행
for cls in D:/project/data/wm-811k/unknown/*/; do
  echo "$cls: $(ls $cls | wc -l)"
done

# 빠른 검증 (생성된 sample 일부만)
python _verify.py --sample 5
```

### 만약 백그라운드 작업이 죽었다면

새 세션이 프로세스 식별자를 알 수 없으므로:
1. 위 카운트로 어디까지 진행됐는지 확인
2. 부족한 분량만 재생성: `python _sample_gen.py --n 200 --workers 8`
   - 기존 파일은 random prefix라 충돌 거의 없음 (filename collision rate ≈ 0)
   - 또는 사용자에게 물어보고 깨끗이 재생성

## 완료된 작업

- [x] WM-811K cca/* 분포 학습 (`_dist_heatmaps/` 8 클래스)
- [x] `_sample_gen.py` v1~v13 micro-tuning 완료
- [x] `_verify.py` 완성
- [x] docs/image-generation/ 5문서
- [x] .claude/skills + agents 3 페어
- [x] memory project_wafer_synthetic_v1.md
- [x] 1 sample/class 테스트 36장 (검증 OK)

## 다음 stage (background 작업 끝난 후)

1. 전체 검증 `python _verify.py` → 7200 / 7200 OK 확인
2. 사용자 시각 spot-check (대표 클래스 1-2개)
3. (다음 stage) contrastive learning 시작
   - `contrastive.py` (이미 repo에 있음, 사용자 제공)
   - 학습 데이터 = 위 7200장
   - HDBSCAN 클러스터링 → centroids
4. (다음 stage) composite map per cluster
5. (다음 stage) evaluation

## Open issues / TODO

- positions JSON FTN/QTN backfill 진행 중이면 `log/_backfill_fq_positions.out` 확인.

## 변경 history

이 STATUS.md 업데이트할 때마다 한 줄 추가:

- 2026-05-01: 7200 background generation 시작, skills/agents/memory 정비, CLAUDE.md 작성
- 2026-05-01: 첫 시도 (optimize=True) ~9% 진행 후 kill. PNG save 병목 → optimize=False + compress_level=1로 재시작 (예상 ~40분, 파일 2배 ≈ 86GB). bg job ID: `bnhiu7d7y`
- 2026-05-01: 생성 완료. 7200/7200 ok=7200 fail=0, 32.5분 소요, rate ~3.7/s. 검증 sample N=5 모두 OK. 디스크: 72GB PNG + 931MB JSON.
- 2026-05-01: positions JSON에 synthetic `partid`/`pgm` + FTN/QTN 추가. `_fq_metadata.py`,
  `_backfill_fq_positions.py` 도입. FTN/QTN hot item은 `b>=200` defect/invalid chip 분포와 맞춰 boost.
