---
description: CNN 학습 실행 (cnn_train.py wrapper)
---

cnn-training skill을 invoke. $ARGUMENTS는 cnn_train.py로 그대로 전달.

예:
- /cnn-train --epochs 30 --batch 16
- /cnn-train --subset-config experiments/quick.yaml --epochs 2

skill이 사전 조건 (data dir, GPU 가용성) 검증 후 명령 실행.
완료 후 결과 폴더 경로 + 다음 step (/cnn-analyze) 안내.
