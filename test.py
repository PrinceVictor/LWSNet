import numpy as np
import paddle
from models.models import *
import paddle.fluid as fluid
from utils.utils import *


if __name__ == "__main__":


    net = Ownnet()

    data_shape = [-1, 1, 368, 1232]

    images = fluid.data(name='disp', shape=data_shape, dtype='float32')

    # print(images.shape)
    # print(unsqueeze(images, [3]).shape)
    # print(images)

    batch_shfit = fluid.layers.expand(fluid.layers.range(-5 + 1, 5, 1, dtype='int32'), [3])
    print(batch_shfit.shape)
    print(batch_shfit)
    batch_shfit = fluid.layers.reshape(batch_shfit, [-1, 1])*3
    batch_shfit = fluid.layers.cast(batch_shfit, 'float32')


    # images = fluid.layers.reshape(images, shape=[-1, 368], inplace=True)
    # print( fluid.layers.reshape(images[:,:,:,:,None], shape=[-1, 368], inplace=True).shape)

    # predict = net.inference(images, images)
    #
    # left_img = np.random.rand(2, 1, 368, 1232).astype(dtype=np.float32)
    # right_img = np.random.rand(2, 1, 368, 1232).astype(dtype=np.float32)
    #
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    # batch_shfit= exe.run(feed={'disp':left_img, 'disp':left_img},
    #                   fetch_list=[batch_shfit])

    batch_shfit = exe.run(fetch_list=batch_shfit)

    print(type(batch_shfit))
    print(batch_shfit[0].shape)
    print(batch_shfit[0].dtype)


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