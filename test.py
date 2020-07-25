import numpy as np
import paddle
from models.models import *
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
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

    place = fluid.CUDAPlace(gpu_id)
    fluid.enable_imperative(place)

    left_img = cv2.imread("/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_2/000004_10.png", cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread("/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_3/000004_10.png", cv2.IMREAD_UNCHANGED)
    left_image = left_img[:, :, ::-1].astype(np.float32)/256*2 - 1
    right_image = right_img[:, :, ::-1].astype(np.float32)/256*2 - 1

    h, w, c = left_img.shape
    th, tw = 368, 1232

    left_image = left_image[h-th:, w-tw:, :]
    right_image = right_image[h-th:, w-tw:, :]

    left_image = left_image.transpose(2, 0, 1)[np.newaxis, :, :, :]
    right_image = right_image.transpose(2, 0, 1)[np.newaxis, :, :, :]

    model = Ownnet(args=args)

    model_state, _ = fluid.dygraph.load_dygraph("results/kitti")
    model.set_dict(model_state)

    model.eval()

    left_image = to_variable(left_image)
    right_image = to_variable(right_image)

    output = model(left_image, right_image)

    for stage in range(stages):
        disp = (output[stage][0][0].numpy()).astype(np.uint8)
        disp = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=1.0, beta=0), cv2.COLORMAP_JET)

        cv2.imshow("disp", disp)
        cv2.imshow("left_img", left_img)
        cv2.waitKey(0)

