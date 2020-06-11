import paddle
import paddle.fluid as fluid

import numpy as np
import argparse

from models.models import *

parser = argparse.ArgumentParser(description='inference img')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
args = parser.parse_args()

def reader_creator_random_image():
    data_shape = [10, 3, 368, 1232]

    def reader():
        for i in range(10):
            yield np.random.uniform(-1, 1, size=3*368*1232).reshape(data_shape), \
                  np.random.uniform(-1, 1, size=3*368*1232).reshape(data_shape)
    return reader

if __name__ == "__main__":

    print("this is dataloader.py")







