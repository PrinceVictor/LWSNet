import paddle
import paddle.fluid as fluid

import numpy as np
import cv2
import os

def img_loader(path):
    img = cv2.imread(path)
    img = img[:,:,::-1].astype(np.float32)
    img = img / 255.0 * 2.0 - 1.0

    return img


def disp_loader(path):
    img = cv2.imread(path)
    img = img.astype(np.float32)
    img = img / 255.0 * 2.0 - 1.0

    return img


class ImageLoad():

    def __init__(self, left, right, disp_l, img_loader=img_loader, disp_loader=disp_loader, training=False):

        self.left = left
        self.right = right
        self.disp_l = disp_l
        self.img_loader = img_loader
        self.disp_loader = disp_loader
        self.training = training

    def create_reader(self):

        data_shape = [1, 3, 368, 1232]

        img_len = len(self.left)
        training = self.training

        def reader():

            for i in range(img_len):
                left = self.left[i]
                right = self.right[i]
                disp_l = self.disp_l[i]

                left_img = self.img_loader(left)
                right_img = self.img_loader(right)
                gt = self.disp_loader(disp_l)

                if training:
                    h, w, c = left_img.shape
                    th, tw = 256, 512

                    x1 = np.random.randint(0, w-tw)
                    y1 = np.random.randint(0, h-th)

                    left_img = left_img[y1:y1+th, x1:x1+tw, :]
                    right_img = right_img[y1:y1+th, x1:x1+tw, :]
                    gt = gt[y1:y1+th, x1:x1+tw, :]

                else:
                    h, w, c = left_img.shape

                    left_img = left_img[h-368:h, w-1232:w, :]
                    right_img = right_img[h-368:h, w-1232:w, :]
                    gt = gt[h-368:h, w-1232:w, :]

                yield left_img, right_img, gt

        return reader

def reader_creator_random_image():
    data_shape = [10, 3, 368, 1232]

    def reader():
        for i in range(10):
            yield np.random.uniform(-1, 1, size=3*368*1232).reshape(data_shape), \
                  np.random.uniform(-1, 1, size=3*368*1232).reshape(data_shape)
    return reader

if __name__ == "__main__":

    print("this is dataloader.py")







