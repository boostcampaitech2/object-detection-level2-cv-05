import os
import random
from PIL import Image
from typing import Tuple
import json
import numpy as np

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
            RandomHorizontalFlip(p=0.5)
        ])

    def __call__(self, image):
        return self.transform(image)

class CustomDataset(Dataset):
    def __init__(self, data_dir='/opt/ml/detection/dataset'):
        self.num_classes = 10
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)
        self.transform = BaseAugmentation(
            resize=(224,224),
            mean=self.mean,
            std=self.std,
        )
        self.val_ratio = 0.2
        JSON_PATH = f'{data_dir}/train.json'

        with open(JSON_PATH, 'r') as json_file:
            train_json = json.load(json_file)

        train_images = train_json['images']
        train_categories = train_json['categories']
        train_annotations = train_json['annotations']
        self.X = list()
        self.y = list()
        for ann in train_annotations:
            file_name = f"{ann['image_id']}".zfill(4) + ".jpg"
            bbox = (ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3])
            self.X.append((file_name, bbox))
            self.y.append(int(ann['category_id']))
        return

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.open(f"/opt/ml/detection/dataset/train/{self.X[idx][0]}")
        img = img.crop(self.X[idx][1])
        X = self.transform(img)
        y = torch.tensor(self.y[idx])
        return X, y

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set 


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths) 


if __name__ == "__main__" :
    dataset = CustomDataset()
    from torch.utils.data import DataLoader
    train_set, val_set = dataset.split_dataset()
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        num_workers=0,
        shuffle=True,
        #pin_memory=use_cuda,
        drop_last=True,
    )
    for idx, train_batch in enumerate(train_loader):
        print(train_batch[0].shape)
        input()