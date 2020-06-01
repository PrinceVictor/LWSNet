import numpy as np
import paddle
from models.models import *
import paddle.fluid as fluid



if __name__ == "__main__":


    net = Ownnet()

    data_shape = [None, 1, 368, 1232]

    images = fluid.data(name='disp', shape=data_shape, dtype='float32')

    predict = net.inference(images, images)

    left_img = np.random.rand(2, 1, 368, 1232).astype(dtype=np.float32)
    right_img = np.random.rand(2, 1, 368, 1232).astype(dtype=np.float32)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    predict = exe.run(feed={'disp':left_img, 'disp':left_img},
                      fetch_list=predict)

    print(predict[-1].shape[0])
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