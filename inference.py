import paddle
from paddle.vision.transforms import Compose, Normalize, ToTensor

import os
import argparse
import glob
import numpy as np
import cv2
import shutil
import time
import PIL.Image as Image

from dataloader.dataloader import imagenet_stats
from models.models import Ownnet
import utils.logger as logger

parser = argparse.ArgumentParser(description='Model Inference')
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--img_path', type=str, default="/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/testing/")
parser.add_argument('--model', type=str, default="results/finetune/checkpoint.pdparams")
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

    if os.path.isdir(args.img_path):
        left_imgs_path = glob.glob(args.img_path + "image_2/*.png")
        right_imgs_path = glob.glob(args.img_path + "image_3/*.png")
    elif os.path.isfile(args.img_path):
        temp_path, img_name = args.img_path.split("/")[0:-2], args.img_path.split("/")[-1]
        temp_path = "/".join(temp_path)
        left_imgs_path = [os.path.join(temp_path, "image_2/"+img_name)]
        right_imgs_path = [os.path.join(temp_path, "image_3/"+img_name)]
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

    inference(model, left_imgs_path, right_imgs_path, LOG)

    LOG.info("End inference!")

def inference(model, left_imgs, right_ims, LOG):

    stages = 4
    model.eval()

    transform = Compose([ToTensor(),
                         Normalize(mean=imagenet_stats["mean"],
                                   std=imagenet_stats["std"])])

    for index in range(len(left_imgs)):

        left_img = cv2.imread(left_imgs[index], cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right_ims[index], cv2.IMREAD_UNCHANGED)

        h, w, c = left_img.shape
        th, tw = 368, 1232

        if h<th or w<tw:
            continue

        left_img = left_img[h - th:h, w - tw:w, :]
        right_img = right_img[h - th:h, w - tw:w, :]

        left_input = transform(left_img[:, :, ::-1]).unsqueeze(axis=0)
        right_input = transform(right_img[:, :, ::-1]).unsqueeze(axis=0)

        with paddle.no_grad():

            start_time = time.time()
            outputs = model(left_input, right_input)
            cost_time = time.time()-start_time
            str = "Inference 4 stages cost = {:.3f} sec, FPS = {:.1f}".format(cost_time, 1/cost_time)

            for stage in range(stages):
                outputs[stage] = outputs[stage].squeeze(axis=[0, 1]).numpy().astype(np.uint8)

                color_disp = cv2.applyColorMap(cv2.convertScaleAbs(outputs[stage], alpha=1, beta=0), cv2.COLORMAP_JET)

            if args.vis:
                concat_img = np.concatenate((left_img, color_disp), axis=0)
                # cv2.imshow("left_img", left_img)
                # cv2.imshow("raw_disp", raw_disp)
                # cv2.imshow("color_disp", color_disp)
                cv2.imshow("concat_img", concat_img)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break

            img_name = left_imgs[index].split("/")[-1]
            save_img_path = os.path.join(args.save_path, img_name)
            cv2.imwrite(save_img_path, color_disp)
            LOG.info("{}\t\tSave img = {}".format(str, save_img_path))




if __name__ == "__main__":

    main()
