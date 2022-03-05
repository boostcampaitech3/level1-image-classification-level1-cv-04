# [AI Tech 3기-level1-P Stage] Image Classification
![https://stages.ai/competitions/104/overview/description](https://user-images.githubusercontent.com/38153357/155888327-a3dc3161-8f4e-44f6-8006-702f6fe91789.png)

## Team 커피사조 (CV-04) ☕

김나영|이승현|이재홍|이현진|전성휴|
:-:|:-:|:-:|:-:|:-:|
[Github](https://github.com/dudskrla)|[Github](https://github.com/sseunghyuns)|[Github](https://github.com/haymrpig)|[Github](https://github.com/leehyeonjin99)|[Github](https://github.com/shhommychon)

## Final Score 🏅

* Public F1 0.7771 → Private F1 0.7564
* Public F1 82.4603 → Private F1 81.4127
* Public F1 3위 → Private F1 7위

<img width="800" alt="image" src="https://user-images.githubusercontent.com/90603530/156859341-1e3dfae1-5d7e-4b10-8878-55bd88256893.png">   

## Time Line 🗓️
<img width = "800" alt="image" src="https://user-images.githubusercontent.com/90603530/156876039-ed4bf244-90e6-490a-b9fc-f65a92ad027d.png">


## Folder Structure 📂
```
level1-cv-04/
│
├── 📝 main.py 
├── 📝 inference.py 
├── 📝 args.py 
│
├── 📂 data/ - default directory for storing input data
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

## Experiments 📈
> Model 실험
> 
- [x]  efficientnet b0~b4 (+ pruned)
- [x]  NFNet
- [x]  SEResnet10
- [x]  vit_base_patch16_384
- [x]  ResNeXt50
- [x]  ResNeXt101
- [x]  RegNet
- [x]  Shufflenet V2

> loss + optimizer 실험
> 
- **Optimizer**
- [x]  Adam
- [x]  AdamP
- [x]  AdamW
- [x]  RAdam
- [x]  madgrad
- **Loss**
- [x]  ce_label_smoothing
- [x]  SymmetricCrossEntropyLoss
- [x]  F1Loss
- [x]  LabelSmoothingLoss
- [x]  ce_f1 loss
- [x]  FocalLoss

> augmentation 실험
> 
- Augmentation
    - [x]  cutmix
    - [x]  mixup
    - [x]  bounding box를 활용한 augmentation
    - [x]  color scale 관련 transforms (Colorjitter , CLAHE 등)


## final model 💻
| Model | Loss | Optimizer | Augmentation | Valid F1 |
| --- | --- | --- | --- | --- |
| efficientnet-b3 | Focal | Adam | Oversampling,Cutmix,CLAHE | 0.8059 |
| regnetx_032 | CE + Label Smoothing Loss | RAdam | CenterCrop,HorizontalFlip,RandomShift | 0.8146 |
| eca_nfnet_l2 | Focal | Adam | CLAHE | 0.8258 |
| resnext50_32x4d | Focal | Adam | CenterCrop,HorizontalFlip,RandomShift | 0.8104 |
| swin transformer | Focal | Adam | CenterCrop,HorizontalFlip,RandomShift | 0.7959 |   
   
   
## Wrap Up Report 📑
[Wrap up.pdf](https://github.com/boostcampaitech3/level1-image-classification-level1-cv-04/files/8190306/Wrap.up.pdf)
