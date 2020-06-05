import numpy as np
import paddle
from models.models import *
import paddle.fluid as fluid
from utils.utils import *
import argparse

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

if __name__ == "__main__":


    net = Ownnet(args=args)

    data_shape = [10, 3, 368, 1232]

    images = fluid.data(name='disp', shape=data_shape, dtype='float32')
    # print(images.shape)
    # print(fluid.layers.reshape(images, shape=[-1,1]).shape)

    # images = fluid.layers.reshape(images, shape=[-1, 368], inplace=True)
    # print( fluid.layers.reshape(images[:,:,:,:,None], shape=[-1, 368], inplace=True).shape)

    predict = net.inference(images, images)
    #
    left_img = np.random.rand(10, 3, 368, 1232).astype(dtype=np.float32)
    right_img = np.random.rand(10, 3, 368, 1232).astype(dtype=np.float32)
    #
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    #
    # a = np.ones((3,4,2), dtype=np.float32)
    # test_a = fluid.layers.ones(shape=[3,4,2], dtype='float32')
    # test_b = fluid.layers.ones(shape=[3,4,2], dtype='float32')
    # test_a = -test_a

    # test_a = fluid.layers.abs(test_a)
    # test = fluid.layers.concat([test_a, test_b], axis=0)
    # test_ = fluid.layers.stack([test_a, test_b], axis=1)
    # test = fluid.layers.reduce_sum(test, dim=1, keep_dim=True)
    # print(type(test))
    # test = fluid.layers.elementwise_sub(test_b[:,:,:1], test_a[:,:,:1])
    # test = test_b[:, :, :] - test_a[:, :, :]



    exe.run(fluid.default_startup_program())

    # predict= exe.run(feed={'disp':left_img, 'disp':left_img},
    #                  fetch_list=[predict])
    # test = exe.run(fetch_list=[test, test_])
    # print(test)
    # print(test[0].shape)
    # print(test[1].shape)

    # batch_shfit = exe.run(fetch_list=batch_shfit)


    # print(predict[-1].shape[0])
    # print(predict[-1])

    # with fluid.dygraph.guard():
    #     temp = fluid.dygraph.base(predict[-1])
    #     print(temp)

    # inference_scope = fluid.core.Scope()

    # with fluid.scope_guard(inference_scope):
    #     [inference_program, feed_target_names,
    #      fetch_targets] = fluid.io.load_inference_model(None, exe)
    #
    #     result = exe.run(inference_program,
    #                      feed={feed_target_names[0]:left_img},
    #                      fetch_list = fetch_targets)