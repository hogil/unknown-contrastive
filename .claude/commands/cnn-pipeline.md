---
description: CNN 학습 → Normal pool predict → threshold 추천 (chained)
---

cnn-pipeline skill을 invoke. 순차 chain:
1. cnn-training agent 호출 ($ARGUMENTS로 학습 옵션)
2. 학습 완료 후 best_model.pth로 cnn-inference 호출
3. Normal pool (`D:/project/data/wm-811k/unknown/Normal/`)에 대한 max_prob 분포 분석
4. threshold 추천 (95% Normal 정확히 잡는 값)

산출: 학습 폴더 + threshold suggestion 보고.
