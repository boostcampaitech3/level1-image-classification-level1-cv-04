# \[AI Tech 3기 Level 1 P Stage\] Image Classification
![https://stages.ai/competitions/104/overview/description](https://user-images.githubusercontent.com/38153357/155888327-a3dc3161-8f4e-44f6-8006-702f6fe91789.png)

## Team 커피사조 (CV-04) ☕

김나영|이승현|이재홍|이현진|전성휴|
:-:|:-:|:-:|:-:|:-:|
[Github](https://github.com/dudskrla)|[Github](https://github.com/sseunghyuns)|[Github](https://github.com/haymrpig)|[Github](https://github.com/leehyeonjin99)|[Github](https://github.com/shhommychon)

## Final Score 🏅

- Public F1 0.7771 → Private F1 **0.7564**
- Public acc 82.4603 → Private acc **81.4127**
- Public 3위 → Private **7위**

<img width="800" alt="image" src="https://user-images.githubusercontent.com/38153357/156905371-3249073f-e5c5-464f-a3dd-f8688baa54e6.gif">

## Wrap Up Report 📑
⭐ [[CV-04] Wrap Up Report.pdf](https://github.com/boostcampaitech3/level1-image-classification-level1-cv-04/files/8190326/CV-04.Wrap.Up.Report.pdf)

<br/>

## Competition Process

### Time Line 🗓️
<img width = "800" alt="image" src="https://user-images.githubusercontent.com/90603530/156876039-ed4bf244-90e6-490a-b9fc-f65a92ad027d.png">

### Experiments 📈
> Model 실험
> 
- [x]  EfficientNet b0~b4 (+ pruned)
- [x]  ECA-NFNet l0~l2
- [x]  ResNet 18~50 
- [x]  SEResNet 50
- [x]  ResNeXt 50/101 32x4d/8d
- [x]  RegNet X/Y 16/32
- [x]  Vision Transformer base/small (patch 16 input size 224)
- [x]  Swin Transformer base/small (patch 4 window 7 input size 224)
- [x]  MNASNet 100
- [x]  MobileNet V3 small
- [x]  Shufflenet V2

> Loss + Optimizer 실험
> 
- **Optimizer**
- [x]  Adam
- [x]  AdamP
- [x]  AdamW
- [x]  RAdam
- [x]  madgrad
- **Loss**
- [x]  Cross Entropy Loss
- [x]  CE + F1 Loss
- [x]  CE + Label Smoothing Loss
- [x]  Symmetric Cross Entropy Loss
- [x]  F1 Loss
- [x]  Focal Loss

> Augmentation 실험
> 
- **color scale transforms**
- [x]  ColorJitter, RandomBrightnessContrast
- [x]  CLAHE
- **shape distortion**
- [x]  Random HorizontalFlip
- [x]  Random Shift
- [x]  Center Crop
- [x]  Resize
- [x]  얼굴 영역 중앙 배치 및 일정/랜덤 확대/축소 ([deepface](https://github.com/serengil/deepface)를 통해 얻은 bbox 정보 사용)
- **image mixing augmentation**
- [x]  CutMix (반반/패치)
- [ ]  Mixup

### Final Model 💻
| Model | Loss | Optimizer | Augmentation | Valid F1 |
| --- | --- | --- | --- | --- |
| efficientnet-b3 | Focal | Adam | CenterCrop, HorizontalFlip, RandomShift,<br/> Oversampling, CutMix, CLAHE | 0.8059 |
| regnetx_032 | CE + Label Smoothing Loss | RAdam | CenterCrop, HorizontalFlip, RandomShift | 0.8146 |
| eca_nfnet_l2 | Focal | Adam | CenterCrop, HorizontalFlip, RandomShift,<br/> CLAHE | 0.8258 |
| resnext50_32x4d | Focal | Adam | CenterCrop, HorizontalFlip, RandomShift | 0.8104 |
| swin_small_patch4_window7_224 | Focal | Adam | CenterCrop, HorizontalFlip, RandomShift | 0.7959 |   

<br/>

## Guidelines to restore our experiments 🔬
### Requirements ⚙
```
(TODO)
torch, numpy, pil, pandas, matplotlib 등 버젼 명시
```

### Folder Structure 📂
```
level1-cv-04/
│
├── 📝 main.py 
├── 📝 inference.py 
├── 📝 args.py 
│
├── 📂 data/ - default directory for storing input data
├── 📂 submission/ - default directory for storing output data
│
├── 📂 model_utils/ - models
│   ├── 📝 model.py
│   └── 📝 custom_module.py
│
├── 📂 trainer/ - trainers
│   ├── 📝 train.py
│   └── 📝 validation.py
│
└── 📂 utils/ - small utility functions
    ├── 📝 util.py
    ├── 📝 augmentation.py
    ├── 📝 dataset.py
    └── 📝 loss.py
```

### Code usage ✒
* (TODO: 아래 내용 수정, 보완 및 정리 예정)

#### 필수적으로 바꿔야 하는 부분

- `utils/util.py` 에서 `wandb_init` 함수의 `wandb.init` 부분
    - 이니셜 수정
- `data/final_train_df.csv` 에서 `img_path` 칼럼 부분을 자신의 환경에 맞게 경로 설정(혹은 `utils/dataset.py`의 데이터로더 부분에서 처리)


#### python, torch 버전

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

#### 학습 실행 예시
```
python main.py --model "efficientnet_b3" --resize 224 224 ; python main.py --model "resnet18" --resize 256 256
```

#### 추론 예시(Single model)
```
python main.py --inference True --save_path "저장된 모델 경로" --model "efficientnet_b3" --resize 224 224
```

