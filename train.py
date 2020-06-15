import paddle
import paddle.fluid as fluid

import argparse
import numpy as np
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

# def train():
#
#     data_shape = [1, 3, 368, 1232]
#
#     net = Ownnet()
#
#     left_image = fluid.data(name='left_img', dtype='float32', shape=data_shape)
#     right_image = fluid.data(name='right_img', dtype='float32', shape=data_shape)
#
#     disp = net.inference(left_image, right_image)

def reader_creator_random_image():
    data_shape = [1, 3, 368, 1232]

    def reader():
        for i in range(10000000000):
            yield np.random.uniform(-1, 1, size=3 * 368 * 1232).reshape(data_shape), \
                  np.random.uniform(-1, 1, size=3 * 368 * 1232).reshape(data_shape)

    return reader

def network():
    data_shape = [5, 3, 368, 1232]

    left_image = fluid.data(name='left_img', dtype='float32', shape=data_shape)
    right_image = fluid.data(name='right_img', dtype='float32', shape=data_shape)

    data_loader = fluid.io.DataLoader.from_generator(feed_list=[left_image, right_image], capacity=10)

    predict = net.inference(left_image, right_image)
    print(len(predict))

    return predict, data_loader


if __name__ == "__main__":

    reader = reader_creator_random_image()
    batch_reader = paddle.batch(reader=reader, batch_size=5)

    for data in batch_reader():
        print(len(data))
        print(data[0][0].shape)



    # net = Ownnet(args)
    #
    #
    #
    # train_prog = fluid.Program()
    # train_startup = fluid.Program()
    #
    #
    # with fluid.program_guard(train_prog, train_startup):
    #     with fluid.unique_name.guard():
    #         disp, train_loader = network()
    #
    # place = fluid.CUDAPlace(0)
    # exe = fluid.Executor(place)
    #
    # exe.run(train_startup)
    #
    # train_loader.set_sample_list_generator(batch_reader, places=fluid.cuda_places(0))
    #
    # for data in train_loader():
    #     predict = exe.run(program=train_prog, feed=data, fetch_list=[disp])
    #     print("success predict", len(disp))
    #     print(predict[0].shape)
    #     print(type(predict[0]))
    #     print(type(disp[0]))
    #     # predict