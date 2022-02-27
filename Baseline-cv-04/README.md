## 필수적으로 바꿔야 하는 부분


- `utils/util.py` 에서 `wandb_init` 함수의 `wandb.init` 부분
- `data/final_train_df.csv` 에서 `img_path` 칼럼 부분을 자신의 환경에 맞게 경로 설정(혹은 `utils/dataset.py`의 데이터로더 부분에서 처리)
- `model_utils/model.py` 에서 실험할 모델 코드 추가
-

## python, torch 버전

- conda 가상환경(선택)

```
conda create -n 가상환경 이름 python=3.8
```

- torch install 
```
pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102
```