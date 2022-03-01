import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceCenterRandomRatioCrop:
    """bbox 정보를 이용하여 얼굴을 중앙으로 위치 시키고 비율에 맞게 zoom in/out 합니다.

    주의사항:
        - sample에는 이미지와 bbox 정보가 포함되어 있어야 합니다.
        - 이 augmentation을 거친 이미즈의 사이즈는 이미지마다 상이합니다.
    """
    def __init__(self, ratio_range=(0.2, 0.4)):
        # 예시: ratio_range가 (0.2, 0.4)일 때, 얼굴의 영역은 전체 이미지의 
        #       20% ~ 40%를 차지하도록 조절됩니다.
        self.ratio_range = ratio_range

    def __call__(self, sample):
        ratio = np.random.uniform(self.ratio_range[0], self.ratio_range[1])
        image, (h, w, x, y) = sample["image"], sample["bbox"]

        # (x, y)는 bbox의 좌측 상단 꼭지점 좌표입니다.
        cen_x, cen_y = round(x + w/2), round(y + h/2)

        face_area = h * w
        img_area = (1-ratio) * face_area / ratio
        img_wh = round(img_area ** 0.5) # img width를 img height와 동일하게 설정합니다.
        img_wh = img_wh + 1 if img_wh % 2 == 1 else img_wh # 짝수로 맞춰줍니다.

        image = np.array(image)
        origin_h, origin_w, _ = image.shape
        move_u, move_d, move_l, move_r = cen_y-img_wh//2, origin_h-(cen_y+img_wh//2), cen_x-img_wh//2, origin_w-(cen_x+img_wh//2)
        # pad or crop up
        if move_u < 0:
            image = np.pad(image, ((-move_u, 0), (0, 0), (0, 0)), "constant", constant_values=0)
        elif move_u > 0:
            image = image[move_u:, :, :]
        # pad or crop down
        if move_d < 0:
            image = np.pad(image, ((0, -move_d), (0, 0), (0, 0)), "constant", constant_values=0)
        elif move_d > 0:
            image = image[:-move_d, :, :]
        # pad or crop left
        if move_l < 0:
            image = np.pad(image, ((0, 0), (-move_l, 0), (0, 0)), "constant", constant_values=0)
        elif move_l > 0:
            image = image[:, move_l:, :]
        # pad or crop right
        if move_r < 0:
            image = np.pad(image, ((0, 0), (0, -move_r), (0, 0)), "constant", constant_values=0)
        elif move_r > 0:
            image = image[:, :-move_r, :]

        return image, ratio


class MinMaxScaleByDivision:
    """torchvisions.transforms.Normalize의 대안

    모든 픽셀을 0에서 1 범위로 정규화 시켜줍니다.
    """
    def __init__(self, div_num=255.):
        self.div_num = div_num
    
    def __call___(self, sample):
        return sample / self.div_num


def cutmix(data, ratio, target=None):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices, :, :, :]
    shuffled_ratio = ratio[indices]
    shuffled_target = target[indices]

    shuffled_data[:, :, :, :shuffled_data.shape[3]//2] = data[:, :, :, :data.shape[3]//2]

    return shuffled_data, ratio, shuffled_ratio, target, shuffled_target


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

def get_train_face_center_crop_transform(args):
    """FaceCenterRandomRatioCrop을 사용할 때의 augmentation 입니다.

    기존 CenterCrop(중앙 크롭)과 ShiftScaleRotate(위치 이동)을 생략합니다.
    """
    return A.Compose([
                    # A.CenterCrop(height=int(512*0.9), width=int(384*0.9), p=1),
                    # A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
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

def pre_transform(args):
    return A.Compose([
                    A.CenterCrop(height=470, width=340, p=1),
                    A.ShiftScaleRotate(scale_limit=0, shift_limit=0.02, rotate_limit=0, p=0.5),
                    A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    ]) 
    
