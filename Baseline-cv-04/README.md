## 필수적으로 바꿔야 하는 부분


- `utils/util.py` 에서 `wandb_init` 함수의 `wandb.init` 부분
- `data/final_train_df.csv` 에서 `img_path` 칼럼 부분을 자신의 환경에 맞게 경로 설정(혹은 `utils/dataset.py`의 데이터로더 부분에서 처리)


## python, torch 버전

- conda 가상환경(선택)

```
conda create -n 가상환경 이름 python=3.8
```

- torch install
```
# conda
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

```
# pip
pip3 install torch torchvision torchaudio
```


## 학습 실행 예시
```
python main.py --model "efficientnet_b3" --resize 224 224 ; python main.py --model "resnet18" --resize 256 256
```

## 추론 예시(Single model)
```
python main.py --inference True --save_path "저장된 모델 경로" --model "efficientnet_b3" --resize 224 224
```
