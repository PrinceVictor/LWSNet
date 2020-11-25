import argparse
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
import paddle.fluid as fluid

import os
import glob
import cv2
import utils.logger as logger
from models.models import *
from dataloader import kitti2015load as kitti
from dataloader import dataloader

parser = argparse.ArgumentParser(description='finetune KITTI')

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
# parser.add_argument('--datapath', default='/home/xjtu/NAS/zhb/dataset/Kitti/data_scene_flow/training/', help='datapath')
parser.add_argument('--datapath', default='/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/training/', help='datapath')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--test_batch_size', type=int, default=4)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--save_path', type=str, default="results/test/")
parser.add_argument('--model', type=str, default="checkpoint")
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--split_file', type=str, default='split.txt')
args = parser.parse_args()

def main():

    LOG = logger.setup_logger(__file__, "./log/")
    for key, value in sorted(vars(args).items()):
        LOG.info(str(key) + ': ' + str(value))

    LOG.info("finetune KITTI main()")

    stages = 4
    gpu_id = args.gpu_id

    place = fluid.CUDAPlace(gpu_id)
    fluid.enable_imperative(place)

    train_left_img, train_right_img, train_left_disp, \
    test_left_img, test_right_img, test_left_disp = kitti.dataloader(args.datapath, args.split_file)

    train_loader = paddle.io.DataLoader(
        dataloader.MyDataloader(train_left_img, train_right_img, train_left_disp, training=True),
        batch_size=args.train_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=True, drop_last=True, num_workers=2)
    test_loader = paddle.io.DataLoader(
        dataloader.MyDataloader(test_left_img, test_right_img, test_left_disp, training=False),
        batch_size=args.test_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=False, drop_last=True, num_workers=2)

    train_batch_len, test_batch_len = len(train_loader), len(test_loader)
    LOG.info("train batch_len {} test batch_len {}".format(train_batch_len, test_batch_len))

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    save_filename = os.path.join(args.save_path, args.model)

    model = Ownnet(args)
    boundaries = [70 * train_batch_len, 150 * train_batch_len]
    values = [args.lr, args.lr * 0.1, args.lr * 0.01]
    optimizer = fluid.optimizer.Adam(learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
                                     parameter_list=model.parameters())

    if args.resume:

        if len(glob.glob(args.resume+"*.pdparams")):
            model_state = paddle.load(glob.glob(args.resume+"*.pdparams")[0])
            model.set_dict(model_state)
        if len(glob.glob(args.resume+"*.pdopt")):
            opt_state = paddle.load(glob.glob(args.resume+"*.pdparams")[0])
            optimizer.set_dict(model_state)
        if len(glob.glob(args.resume+"*.params")):
            param_state = paddle.load(glob.glob(args.resume+"*.pdparams")[0])
            model.set_dict(model_state)

    epoch = 0

    # checkpoint = paddle.load(save_filename + ".param")
    # print(checkpoint)

    error_3pixel_check = np.inf

    for epoch in range(args.epoch):
        stage_loss_list = np.zeros((4), dtype=np.float32)
        sum_losses_rec = 0
        error_3pixel_list = np.zeros((4), dtype=np.float32)
        error_3pixel = 0

        model.train()

        for batch_id, data in enumerate(train_loader()):
            left_img, right_img, gt = data

            mask = paddle.to_tensor(gt.numpy() > 0)
            gt_mask = paddle.masked_select(gt, mask)

            outputs = model(left_img, right_img)
            outputs = [paddle.squeeze(output) for output in outputs]

            tem_stage_loss = []
            for index in range(stages):

                temp_loss = args.loss_weights[index]* F.smooth_l1_loss(paddle.masked_select(outputs[index], mask), gt_mask, reduction='mean')
                tem_stage_loss.append(temp_loss)
                stage_loss_list[index] += temp_loss.numpy()

            sum_loss = fluid.layers.sum(tem_stage_loss)
            # sum_loss = tem_stage_loss[3]
            sum_loss.backward()
            optimizer.minimize(sum_loss)
            model.clear_gradients()

            sum_losses_rec += sum_loss.numpy()

            info_str = ['Stage {} = {:.2f}'.format(x, float(stage_loss_list[x]/ (batch_id + 1))) for x in range(stages)]
            info_str = '\t'.join(info_str)

            print("Train: epoch {}, batch_id {} learn_lr {:.5f} sum_loss {} \t {}".format(epoch,
                                                                                          batch_id,
                                                                                          optimizer.current_step_lr(),
                                                                                          np.round(sum_losses_rec / (batch_id + 1),3),
                                                                                          info_str))


        with fluid.dygraph.no_grad():
            model.eval()

            for batch_id, data in enumerate(test_loader()):
                left_img, right_img, gt = data

                outputs = model(left_img, right_img)
                outputs = [paddle.squeeze(output) for output in outputs]

                for stage in range(stages):
                    error_3pixel_list[stage] += error_estimating(outputs[stage].numpy(), gt.numpy())
                    error_3pixel = sum(error_3pixel_list)

                info_str = ['Stage {} = {:.2f}'.format(x, float(error_3pixel_list[x] / (batch_id + 1))) for x in range(stages)]
                info_str = '\t'.join(info_str)

                print("Test: epoch {}, batch_id {} error 3pixel {}\t  {}" .format(epoch,
                                                                                  batch_id,
                                                                                  round(error_3pixel/(batch_id+1), 4),
                                                                                  info_str))

                # for batch_size_I in range(args.test_batch_size):
                #     disp = (output[3][batch_size_I][0].numpy()).astype(np.uint8)
                #     disp = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=1.0, beta=0), cv2.COLORMAP_JET)
                #
                #     gt_disp = (gt[batch_size_I][0].numpy()).astype(np.uint8)
                #     gt_disp = cv2.applyColorMap(cv2.convertScaleAbs(gt_disp, alpha=1.0, beta=0), cv2.COLORMAP_JET)
                #     cv2.imshow("disp", disp)
                #     cv2.imshow("gt_disp", gt_disp)
                #     cv2.waitKey(0)
                #
                # raise StopIteration

            if error_3pixel / (batch_id + 1) < error_3pixel_check:
                error_3pixel_check = error_3pixel / (batch_id + 1)

                # paddle.save(model.state_dict(), save_filename+".pdparams")
                # paddle.save(optimizer.state_dict(), save_filename+".pdopt")
                # paddle.save({"epoch":epoch, "error":3}, save_filename + ".param")
                print("save model param success")


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    # print(disp.shape, ground_truth.shape, np.max(gt), np.min(gt))
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = np.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return float(err3) / float(mask.sum())


if __name__ == "__main__":

    main()


