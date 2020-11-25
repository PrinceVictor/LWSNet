import argparse
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
from paddle.io import DataLoader

import numpy as np
import os
import glob
import math

from models.models import *
from dataloader import sceneflow as sf
from dataloader import dataloader
import utils.logger as logger
from utils.utils import AverageMeter as AverageMeter

parser = argparse.ArgumentParser(description='pretrain Sceneflow main()')

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/home/xjtu/NAS/zhb/dataset/sceneflow/')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--last_epoch', type=int, default=-1)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--save_path', type=str, default="results/pretrained/")
parser.add_argument('--model', type=str, default="checkpoint")
parser.add_argument('--resume', type=str, default="")
args = parser.parse_args()

def main():

    LOG = logger.setup_logger(__file__, "./log/")
    for key, value in vars(args).items():
        LOG.info(str(key) + ': ' + str(value))

    LOG.info("pretrain Sceneflow main()")

    stages = 4
    gpu_id = args.gpu_id

    place = fluid.CUDAPlace(gpu_id)
    fluid.enable_imperative(place)

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = sf.dataloader(args.datapath)

    train_loader = paddle.io.DataLoader(
        dataloader.MyDataloader(train_left_img, train_right_img, train_left_disp, training=True, kitti_set=False),
        batch_size=args.train_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=True, drop_last=False, num_workers=2)
    test_loader = paddle.io.DataLoader(
        dataloader.MyDataloader(test_left_img, test_right_img, test_left_disp, training=False, kitti_set=False),
        batch_size=args.test_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=False, drop_last=False, num_workers=2)

    train_batch_len, test_batch_len = len(train_loader), len(test_loader)
    LOG.info("train batch_len {} test batch_len {}".format(train_batch_len, test_batch_len))

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    save_filename = os.path.join(args.save_path, args.model)

    model = Ownnet(args)

    last_epoch = 0
    error_check = math.inf

    boundaries = [150*train_batch_len, 150*train_batch_len, 300*train_batch_len]
    values = [args.lr, args.lr * 0.1, args.lr * 0.01, args.lr * 0.001]
    # optimizer = fluid.optimizer.Adam(learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
    #                             parameter_list=model.parameters())
    optimizer = fluid.optimizer.Adam(learning_rate=args.lr,
                                     parameter_list=model.parameters())

    if args.resume:
        if len(glob.glob(args.resume + "*.pdparams")):
            model_state = paddle.load(glob.glob(args.resume + "*.pdparams")[0])
            model.set_dict(model_state)
            LOG.info("load model state")

        if len(glob.glob(args.resume + "*.pdopt")):
            opt_state = paddle.load(glob.glob(args.resume + "*.pdopt")[0])
            optimizer.set_dict(opt_state)
            LOG.info("load optimizer state")

        if len(glob.glob(args.resume + "*.params")):
            param_state = paddle.load(glob.glob(args.resume + "*.params")[0])
            last_epoch = param_state["epoch"] + 1
            error_check = param_state["error"]
            LOG.info("load last epoch and error")

        LOG.info("resume successfully")

    if args.last_epoch != -1:
        last_epoch = args.last_epoch

    for epoch in range(last_epoch, args.epoch):

        losses = [AverageMeter() for _ in range(stages)]
        model.train()

        for batch_id, data in enumerate(train_loader()):
            left_img, right_img, gt = data

            mask = paddle.to_tensor(gt.numpy() < args.maxdisp)
            gt_mask = paddle.masked_select(gt, mask)
            if paddle.cast(mask, "float32").sum() == 0:
                continue

            outputs = model(left_img, right_img)
            outputs = [paddle.squeeze(output) for output in outputs]

            stage_loss = []
            for index in range(stages):
                loss = args.loss_weights[index] * F.smooth_l1_loss(paddle.masked_select(outputs[index], mask),
                                                                   gt_mask, reduction='mean')
                stage_loss.append(loss)
                losses[index].update(float(loss.numpy())/args.loss_weights[index])

            sum_loss = fluid.layers.sum(stage_loss)
            sum_loss.backward()
            optimizer.minimize(sum_loss)
            model.clear_gradients()

            if batch_id % 5 == 0:
                info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
                info_str = '\t'.join(info_str)

                LOG.info('Epoch{} [{}/{}]  lr:{:.5f}\t{}'.format(epoch, batch_id, train_batch_len, optimizer.current_step_lr(), info_str))

        info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
        LOG.info('Average train loss = ' + info_str)

        EPEs = [AverageMeter() for _ in range(stages)]
        model.eval()

        for batch_id, data in enumerate(test_loader()):
            left_img, right_img, gt = data

            gt = gt.numpy()
            mask = gt < args.maxdisp

            with fluid.dygraph.no_grad():
                outputs = model(left_img, right_img)

                for stage in range(stages):
                    if len(gt[mask]) == 0:
                        continue
                    output = paddle.squeeze(outputs[stage], 1).numpy()
                    output = output[:, 4:, :]
                    EPEs[stage].update(np.mean(np.abs(output[mask], gt[mask])))

            info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
            print('Test: [{}/{}] {}'.format(batch_id, test_batch_len, info_str))

        info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
        print('Average test EPE = ' + info_str)

        if EPEs[-1].avg < error_check:
            error_check = EPEs[-1].avg

            paddle.save(model.state_dict(), save_filename + ".pdparams")
            paddle.save(optimizer.state_dict(), save_filename + ".pdopt")
            paddle.save({"epoch": epoch, "error": error_check}, save_filename + ".params")
            LOG.info("save model param success")


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    # print(disp.shape, ground_truth.shape, np.max(gt), np.min(gt))
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = np.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return float(err3) / float(mask.sum() + 1e-9)


if __name__ == "__main__":
    main()





