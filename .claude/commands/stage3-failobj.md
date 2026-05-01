---
description: 3-stage CNN training (chip → obj_id maps → 3-channel wafer CNN). Dispatches stage3-failobj agent.
---

# /stage3-failobj

stage3-failobj agent를 호출해 3단계 학습을 일관되게 진행합니다:

1. **Stage 1** — chip 5-class 분류기 (`cnn_train.py --data-dir classification_chips`)
2. **Stage 2** — chip 분류기 inference로 wafer별 32×32 obj_id .npy cache (`_build_obj_id_maps.py`)
3. **Stage 3** — 3-channel feature wafer CNN (`cnn_train_failobj.py`)

agent가 자원/산출물 검증, GPU 경합 점검, optional `--init-from` (TAPT) 결정,
실패 시 abort + 산출 보존을 처리합니다.

상세 정책 + OBJECT_TYPE_ID 매핑 + BICUBIC G channel 의 근거는
`.claude/skills/stage3-failobj/SKILL.md` 참고.
