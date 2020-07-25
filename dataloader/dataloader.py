import paddle
import paddle.fluid as fluid

import numpy as np
import cv2
import os

from PIL import Image
from . import readpfm as rp
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)

def disparity_loader2(path):
    return Image.open(path)

def img_loader(path):
    img = cv2.imread(path)
    img = img[:,:,::-1].astype(np.float32)
    img = img / 256.0 * 2.0 - 1.0
    return img

def disp_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img / 256.0
    return img

class sceneflow_dataloader():
    def __init__(self, left, right, left_disparity, training=True, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def create_reader(self):

        img_len = len(self.left)
        training = self.training

        def reader():

            for i in range(img_len):
                left = self.left[i]
                right = self.right[i]
                disp_L = self.disp_L[i]

                left_img = self.loader(left)
                right_img = self.loader(right)
                dataL, scaleL = self.dploader(disp_L)
                gt = np.ascontiguousarray(dataL, dtype=np.float32)

                if training:
                    w, h = left_img.size
                    th, tw = 256, 512
                    # th, tw = 540, 960

                    x1 = np.random.randint(0, w - tw)
                    y1 = np.random.randint(0, h - th)

                    left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                    right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
                    gt = gt[y1:y1 + th, x1:x1 + tw]

                else:
                    w, h = left_img.size
                    left_img = left_img.crop((w - 512, h - 256, w, h))
                    right_img = right_img.crop((w - 512, h - 256, w, h))


                left_img = np.ascontiguousarray(left_img, dtype=np.float32) / 256 * 2 - 1.0
                right_img = np.ascontiguousarray(right_img, dtype=np.float32) / 256 * 2 - 1.0

                left_img = left_img.transpose(2, 0, 1)
                right_img = right_img.transpose(2, 0, 1)
                gt = gt.transpose(0, 1)[np.newaxis, :, :]

                yield left_img, right_img, gt

        return reader



class ImageLoad():

    def __init__(self, left, right, disp_l, img_loader=default_loader, disp_loader=disparity_loader2, training=False):

        self.left = left
        self.right = right
        self.disp_l = disp_l
        self.img_loader = img_loader
        self.disp_loader = disp_loader
        self.training = training

    def create_reader(self):

        img_len = len(self.left)
        training = self.training

        def reader():

            for i in range(img_len):
                left = self.left[i]
                right = self.right[i]
                disp_l = self.disp_l[i]
                # print(left)

                left_img = self.img_loader(left)
                right_img = self.img_loader(right)
                gt = self.disp_loader(disp_l)
                # print('gt_shape', gt.shape)

                if training:
                    w, h = left_img.size
                    th, tw = 256, 512

                    x1 = np.random.randint(0, w - tw)
                    y1 = np.random.randint(0, h - th)

                    left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                    right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

                    gt = np.ascontiguousarray(gt, dtype=np.float32) / 256
                    gt = gt[y1:y1 + th, x1:x1 + tw]

                else:
                    w, h = left_img.size

                    left_img = left_img.crop((w - 1232, h - 368, w, h))
                    right_img = right_img.crop((w - 1232, h - 368, w, h))
                    w1, h1 = left_img.size

                    gt = gt.crop((w - 1232, h - 368, w, h))
                    gt = np.ascontiguousarray(gt, dtype=np.float32) / 256

                left_img = np.ascontiguousarray(left_img, dtype=np.float32) / 256 * 2 - 1.0
                right_img = np.ascontiguousarray(right_img, dtype=np.float32) / 256 * 2 - 1.0

                left_img = left_img.transpose(2, 0, 1)
                right_img = right_img.transpose(2, 0, 1)
                gt = gt.transpose(0, 1)[np.newaxis,:,:]

                yield left_img, right_img, gt

        return reader

if __name__ == "__main__":

    print("this is dataloader.py")







