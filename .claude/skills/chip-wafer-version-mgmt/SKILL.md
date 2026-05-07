---
name: chip-wafer-version-mgmt
description: chip + wafer 합성 version 관리 — 새 version rollout / 특정 version 복원 / preview 비교 / 양 repo (known-cnn + unknown-contrastive) sync 가이드. 매 합성 spec 변경 시 사용.
---

# chip-wafer-version-mgmt skill

chip + wafer 합성 의 version (v0, v1, ..., v5, v5.1, ...) 관리 entry point.

## 항상 먼저 읽을 doc

| 위치 | 역할 |
|---|---|
| `docs/chip-multilabel/VERSION_HISTORY.md` | 모든 version 통합 표 (commit / spec / backup / preview) |
| `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md` | 현재 canonical spec + history |
| `docs/chip-multilabel/V5_REGEN_MANIFEST.md` | 본 합성 manifest |

## 양 repo 동시 관리 (mirror 정책)

3 chip 합성 파일은 **반드시 byte-identical**:
- `dist_apply/_synth_chips_only.py` (known-cnn) ↔ `_synth_chips_only.py` (unknown-contrastive root)
- `dist_apply/_sample_gen.py` ↔ root
- `dist_apply/_sample_gen_gpu.py` ↔ root

모든 변경:
1. known-cnn 에서 먼저 patch
2. `cp -f` 로 unknown-contrastive 미러
3. `diff -q` 로 검증
4. 양 repo 동시 commit + push

## Quick reference

### A) 새 version 만들기

1. **Code patch** — 3 파일 (known-cnn) 변경
2. **Mirror** — `cp -f known-cnn/dist_apply/_*.py unknown-contrastive/`
3. **Sample 생성** — 6-12 chip + 4-6 wafer preview, 사용자 approve
4. **(major 만) Backup** — `mv classification_chips classification_chips_pre_v<X>_<YYMMDD>` 등 4 폴더
5. **본 합성** — 각 명령:
   ```bash
   # chip 1000
   python dist_apply/_synth_chips_only.py --per-class 200 --out D:/project/data/wm-811k/classification_chips
   # chip_multilabel master 2250
   python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel \
       --per-defect 200 --per-normal 200 --per-invalid 50 --source-strength-pct 50 --seed 42
   # wafer 8000 (obj-active 32 + wafer-canvas 8)
   python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8
   # 또는 obj-active only (fast):
   python dist_apply/_sample_gen.py --n 200 --workers 6
   ```
6. **VERSION_HISTORY.md 업데이트** — new tag row 추가
7. **CHIP_SYNTH_V5_SPEC.md history 표 업데이트** (새 version 추가)
8. **commit + push 양 repo** (commit prefix: `v<tag>:`)

### B) 특정 version 복원 (rollback)

#### B-1) 코드만

```bash
COMMIT_HASH=<hash>   # VERSION_HISTORY.md 표에서 찾기
git checkout $COMMIT_HASH -- dist_apply/_synth_chips_only.py dist_apply/_sample_gen.py dist_apply/_sample_gen_gpu.py
cp -f dist_apply/_*.py /d/project/unknown-contrastive/
```

#### B-2) 데이터까지

```bash
# 1) 현재 active 안전 보존
mv D:/project/data/wm-811k/{classification_chips,chip_multilabel,unknown} D:/project/data/wm-811k/<each>_pre_rollback_<YYMMDD>
mv D:/project/data/positions/unknown D:/project/data/positions/unknown_pre_rollback_<YYMMDD>

# 2) backup → active 또는 새 합성
mv D:/project/data/wm-811k/classification_chips_pre_v5_260507 D:/project/data/wm-811k/classification_chips
# ... 4 folders
```

#### B-3) 새 합성 (백업 없을 때)

코드 checkout 후 Quick A의 5번 명령 실행.

### C) Preview 비교

| 위치 | 내용 |
|---|---|
| `_chip_revert_preview/d70daaf_RGB_Normal_Invalid/` | v0 RGB Normal/Invalid |
| `_chip_revert_preview/d8ab78d_palette_beta_Normal_Invalid/` | v1 palette+Beta |
| `_chip_revert_preview/synth_v20_fork_scratch_etc/` | 현재 chip class |
| `_wafer_v5_preview/` | v5 wafer 6400×6400 |
| `_fork_v5_1_sample/` | v5.1 fork |

같은 위치 폴더명 일관 — 새 version 도 `_<tag>_<info>_preview/` 패턴 따라야.

### D) 절대 규칙

1. **이미지 생성 후 출력 폴더 절대 경로 즉시 표시** (`CLAUDE.md` 절대 규칙).
2. **양 repo 동시 commit/push** — 중간 push 금지.
3. **VERSION_HISTORY.md + CHIP_SYNTH_V5_SPEC.md 동시 업데이트** — partial update 금지.
4. **backup 폴더 무단 삭제 금지** — 사용자 명시 요청 전.
5. **palette PNG only** (mode='P', grade 0-7). RGB 자유 색 영구 금지.
6. **rotation/flip aug 영구 금지** + **TTA 영구 금지** (chip multi-label).

### E) Agent 자동화

`.claude/agents/chip-wafer-regen.md` agent 가:
- 사용자 spec param 받음 (예: "fork 0.55/0.91 로 v5.2")
- 3 파일 patch + 양 repo mirror + diff 검증
- sample 생성 + 사용자 visual approve 대기
- 본 합성 명령 dispatch (background)
- VERSION_HISTORY.md / CHIP_SYNTH_V5_SPEC.md 업데이트
- commit + push 양 repo

복잡한 multi-step 변경 시 agent 호출. 단일 param tweak 은 직접 patch.
