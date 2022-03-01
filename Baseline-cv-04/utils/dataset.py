import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd

from utils.augmentation import FaceCenterRandomRatioCrop

class MaskDataset(Dataset):
    def __init__(self, dataframe, transform=None, multi_output=False):
        super().__init__()
        self.img_path = dataframe["img_path"].values
        self.label = dataframe["all"].values
        self.multi_output = multi_output
        if self.multi_output:
            self.mask_label = dataframe["mask"].values
            self.gender_label = dataframe["gender"].values
            self.age_label = dataframe["age_group"].values
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path[index]
        label = self.label[index]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        
        if not self.multi_output:
            return img, label
        else:
            return img, {
                "mask": self.mask_label[index],
                "gender": self.gender_label[index],
                "age": self.age_label[index],
                "ans": label
            }

    def __len__(self):
        return len(self.label)


class MaskFaceCenterDataset(Dataset):
    """utils/augmentation/FaceCenterRandomRatioCrop을 사용할 때의 Dataset
    """
    def __init__(self, dataframe, transform=None, multi_output=False):
        super().__init__()
        self.img_path = dataframe["img_path"].values
        self.bbox = dataframe[[
            "deepface_bbox_h", "deepface_bbox_w", "deepface_bbox_x", "deepface_bbox_y"
        ]].values
        self.label = dataframe["all"].values
        self.multi_output = multi_output
        if self.multi_output:
            self.mask_label = dataframe["mask"].values
            self.gender_label = dataframe["gender"].values
            self.age_label = dataframe["age_group"].values
        self.face_center_crop = FaceCenterRandomRatioCrop((0.2, 0.4))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path[index]
        label = self.label[index]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            h, w, x, y = self.bbox[index, :]
            img, face_area_ratio = self.face_center_crop({"image": img, "bbox": (h, w, x, y)})
            img = self.transform(image=img)["image"]

        if not self.multi_output:
            return { "image": img, "ratio": face_area_ratio }, label
        else:
            return { "image": img, "ratio": face_area_ratio }, {
                "mask": self.mask_label[index],
                "gender": self.gender_label[index],
                "age": self.age_label[index],
                "ans": label
            }

    def __len__(self):
        return len(self.label)


class MaskTestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.id = dataframe["ImageID"].values
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread("/opt/ml/input/data/eval/images/" + self.id[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        
        return img, self.id[index]


    def __len__(self):
        return len(self.id)
