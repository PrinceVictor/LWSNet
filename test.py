import paddle
from paddle.vision.transforms import Compose, Normalize, Transpose

import os
import argparse
import glob
import numpy as np
import cv2
import shutil
import time

from dataloader.dataloader import imagenet_stats
from models.models import Ownnet
import utils.logger as logger

parser = argparse.ArgumentParser(description='Model Inference')
parser.add_argument('--max_disparity', type=int, default=192)
# parser.add_argument('--left_img_path', type=str, default="/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_2")
# parser.add_argument('--right_img_path', type=str, default="/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_3")
parser.add_argument('--left_img_path', type=str,
                    default="/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_2/000004_10.png")
parser.add_argument('--right_img_path', type=str,
                    default="/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/image_3/000004_10.png")
parser.add_argument('--model', type=str, default="result/finetune/checkpoint.params")
parser.add_argument('--save_path', type=str, default="results/inference")
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--vis', action='store_true', default=False, help="Show inference results")
args = parser.parse_args()

def main():

    LOG = logger.setup_logger(__file__, "./log/")
    for key, value in vars(args).items():
        LOG.info(str(key) + ': ' + str(value))

    gpu_id = args.gpu_id
    place = paddle.set_device("gpu:" + str(gpu_id))

    if os.path.isdir(args.left_img_path):
        left_imgs_path = glob.glob(args.left_img_path + "/*.png")
        right_imgs_path = glob.glob(args.right_img_path + "/*.png")
    else:
        left_imgs_path = [args.left_img_path]
        right_imgs_path = [args.right_img_path]
    LOG.info("Load data path")

    model = Ownnet(args)
    if not os.path.isfile(args.model):
        LOG.info("No model load")
        raise SystemExit
    else:
        model.set_state_dict(paddle.load(args.model))
        LOG.info("Successful load model")

    model.eval()

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)
    LOG.info("Clear all files in the path: {}".format(args.save_path))

    LOG.info("Begin inference!")

    inference(model, left_imgs_path, right_imgs_path)

    LOG.info("End inference!")






def inference(model, left_imgs, right_ims):

    stages = 4
    model.eval()

    transform = Compose([Transpose(),
                         Normalize(mean=imagenet_stats["mean"],
                                   std=imagenet_stats["std"])])

    for index in range(len(left_imgs)):
        left_img = cv2.imread(left_imgs[index], cv2.IMREAD_UNCHANGED)[::-1]
        right_img = cv2.imread(right_ims[index], cv2.IMREAD_UNCHANGED)[::-1]

        print(np.transpose(left_img, (2, 0, 1)))

        left_img = paddle.to_tensor(transform(left_img)).unsqueeze(axis=0)
        right_img = paddle.to_tensor(transform(right_img)).unsqueeze(axis=0)

        print(left_img)

        # output = model(left_image, right_image)
        #
        # for stage in range(stages):
        #     disp = (output[stage][0][0].numpy()).astype(np.uint8)
        #     # cv2.normalize(disp, disp, 0, 256, cv2.NORM_MINMAX)
        #     disp = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=1.0, beta=0), cv2.COLORMAP_JET)
        #
        #     cv2.imshow("disp", disp)
        #     cv2.imshow("left_img", left_img)
        #     cv2.waitKey(0)



if __name__ == "__main__":

    main()
