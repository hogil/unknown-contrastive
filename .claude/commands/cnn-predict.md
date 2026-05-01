---
description: CNN 예측 (cnn_predict.py wrapper, threshold sweep)
---

cnn-inference skill을 invoke. $ARGUMENTS는 cnn_predict.py로 전달.

예:
- /cnn-predict --model log/<run>/best_model.pth --input D:/test_dir --output preds.json
- /cnn-predict --model best_model.pth --input <dir> --threshold 0.7
- /cnn-predict --model best_model.pth --input val_dir --threshold-sweep 0.1,0.9,0.05

threshold-sweep 시 label 추정 가능한 폴더 구조 (`{class}/img.png`) 권장.
