---
description: 3-stage compound CNN training (chip → obj_id maps → compound 3-channel wafer CNN). Dispatches stage3-compound agent.
---

# /stage3-compound

stage3-compound agent를 호출해 3단계 학습을 일관되게 진행합니다:

1. **Stage 1** — chip 5-class 분류기 (`cnn_train.py --data-dir classification_chips`) → `logs_chip/`
2. **Stage 2** — chip 분류기 inference로 wafer별 32×32 obj_id .npy cache (`_build_obj_id_maps.py`)
3. **Stage 3** — 3-channel feature compound 분류기 (`cnn_train_compound.py`) → `logs_compound/`

각 logs_*/ 안에 `overall/` 폴더에 val_f1 best run 통째 복사 자동 갱신.

agent가 자원/산출물 검증, GPU 경합 점검, optional `--init-from` (TAPT) 결정,
실패 시 abort + 산출 보존을 처리합니다.

상세 정책 + obj_id 매핑 + BICUBIC G channel 근거 + `_meta.json` 컨벤션은
`.claude/skills/stage3-compound/SKILL.md` 참고.
