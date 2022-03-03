import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


##### get train/valid transform #####

def get_train_transform(args):
    return A.Compose([
                    A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
                    A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    A.CLAHE(),
                    A.Normalize(
                        mean=[0.56, 0.524, 0.501],
                        std=[0.258, 0.265, 0.267],
                        max_pixel_value=255.0),
                    ToTensorV2()
                    ])

def get_valid_transform(args):
    return A.Compose([
                    A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
                    A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    A.Normalize(
                        mean=[0.56, 0.524, 0.501],
                        std=[0.258, 0.265, 0.267],
                        max_pixel_value=255.0),
                    ToTensorV2()
                    ])
    

def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2