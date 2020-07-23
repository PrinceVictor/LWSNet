import numpy as np
import paddle
from models.models import *
import paddle.fluid as fluid
from utils.utils import *
import argparse
import cv2

parser = argparse.ArgumentParser(description='inference img')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":

    stages = 4
    gpu_id = args.gpu_id

    net = Ownnet(args=args)

    data_shape = [4, 3, 256, 512]

    main_pro = fluid.Program()
    startup_prog = fluid.Program()

    # with fluid.program_guard(main_pro, startup_prog):
    #
    #     left_image = fluid.data(name='left_img', dtype='float32', shape=data_shape)
    #     right_image = fluid.data(name='right_img', dtype='float32', shape=data_shape)
    #
    #     predict = net.inference(left_image, right_image)
    #     # print(len(predict), predict[0].shape)
    #     predict_ouput = fluid.layers.stack(x=predict, axis=0)
        # print(predict_ouput.shape)

    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    [inference_program, feed_target_names, fetch_targets] = \
        (fluid.io.load_inference_model(dirname="results/inference", executor=exe))

    left_img = cv2.imread("/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_2/000004_10.png", cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread("/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_3/000004_10.png", cv2.IMREAD_UNCHANGED)
    left_image = left_img[:, :, ::-1].astype(np.float32)
    right_image = right_img[:, :, ::-1].astype(np.float32)
    h, w, c = left_img.shape

    th, tw = 256, 512

    x1 = np.random.randint(0, w - tw)
    y1 = np.random.randint(0, h - th)
    left_image = left_image[y1:y1 + th, x1:x1 + tw, :]
    right_image = right_image[y1:y1 + th, x1:x1 + tw, :]

    left_image = left_image.transpose(2, 0, 1)[np.newaxis, :, :, :]
    right_image = right_image.transpose(2, 0, 1)[np.newaxis, :, :, :]

    left_image = np.tile(left_image, (4, 1, 1, 1))
    right_image = np.tile(right_image, (4, 1, 1, 1))


    results = exe.run(program=inference_program,
                      feed={feed_target_names[0]:left_image, feed_target_names[1]:right_image},
                      fetch_list=fetch_targets)
    staget = 4
    disp = results[0][stages -1][3][0]
    #
    disp = (disp).astype(np.uint8)
    cv2.imshow("disp", disp)
    cv2.imshow("left_img", left_img)
    cv2.waitKey(0)

