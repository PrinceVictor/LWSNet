import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Normalize, Transpose
import os
from PIL import Image
from . import readpfm as rp
import numpy as np
import random

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def default_loader(path):
    return Image.open(path).convert('RGB')

def sceneflow_dispLoader(path):
    return rp.readPFM(path)

def Kitti_dispLoader(path):
    return Image.open(path)

class MyDataloader(Dataset):
    def __init__(self, left, right, left_disparity, training=True, loader=default_loader, kitti_set=True):
        super(MyDataloader, self).__init__()

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.training = training
        self.kitti_set = kitti_set
        if self.kitti_set:
            self.dploader = Kitti_dispLoader
        else:
            self.dploader = sceneflow_dispLoader

        self.tramsform = Compose([Transpose(),
                                  Normalize(mean=imagenet_stats["mean"], std=imagenet_stats["std"])])


    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        if self.kitti_set:
            dataL = self.dploader(disp_L)
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
        else:
            dataL, scaleL = self.dploader(disp_L)
            dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img = self.tramsform(left_img)
            right_img = self.tramsform(right_img)

            return left_img, right_img, dataL

        else:
            w, h = left_img.size

            if self.kitti_set:
                left_img = left_img.crop((w - 1232, h - 368, w, h))
                right_img = right_img.crop((w - 1232, h - 368, w, h))
                dataL = dataL[h - 368:h, w-1232:w]
            else:
                left_img = left_img.crop((w - 960, h - 544, w, h))
                right_img = right_img.crop((w - 960, h - 544, w, h))
                dataL = dataL[h - 544:h, w - 960:w]

            left_img = self.tramsform(left_img)
            right_img = self.tramsform(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

if __name__ == "__main__":

    print("this is dataloader.py")