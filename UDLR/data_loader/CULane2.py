import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image


class CULane(Dataset):
    def __init__(self, path, image_set, size, theta=None):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.size = size
        self.theta = theta
        self.transform = Compose([ToTensor()])

        if image_set != 'test':
            self.createIndex()
        else:
            self.createIndex_test()

    def createIndex(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}_gt.txt".format(self.image_set))

        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def createIndex_test(self):
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))

        self.img_list = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, line[1:]))

    # def rotate(self, img):
    #     u = np.random.uniform()
    #     degree = (u-0.5) * self.theta
    #     R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
    #     img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    #     return img

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.size, Image.BICUBIC)
        img = self.transform(img)

        if not os.path.exists(self.img_list[idx]):
            print(self.img_list[idx])
        if self.image_set != 'test':
            # img = self.rotate(img)
            segLabel = Image.open(self.segLabel_list[idx])
            segLabel = segLabel.resize(self.size, Image.NEAREST)
            segLabel = self.transform(segLabel)[0] * 255
            # segLabel = self.rotate(segLabel)
            exist = np.array(self.exist_list[idx])
            exist = torch.from_numpy(exist).type(torch.float32)
        else:
            segLabel = None
            exist = None

        sample = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': self.img_list[idx]}
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