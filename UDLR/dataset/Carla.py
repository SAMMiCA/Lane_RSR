import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image


class Carla(Dataset):
    def __init__(self, path, transform):
        super(Carla, self).__init__()
        self.data_dir_path = path
        self.transform = transform
        self.img_list = sorted(os.listdir(self.data_dir_path))

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_dir_path, self.img_list[idx]))

        segLabel = None
        exist = None

        sample = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': os.path.join(self.data_dir_path, self.img_list[idx])}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': [x['img_name'] for x in batch]}

        return samples


if __name__ == '__main__':
    path = '../../Carla/'
    size = (256, 256)
    carla = Carla(path, size)
