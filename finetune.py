import argparse
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle.io import DataLoader

import cv2
from PIL import Image

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
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=4)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default="test")
parser.add_argument('--resume', action='store_true', default=False,)

args = parser.parse_args()

def main():

    print("finetune KITTI main()")

    stages = 4
    gpu_id = args.gpu_id

    place = paddle.set_device("gpu:"+str(gpu_id))

    train_left_img, train_right_img, train_left_disp, \
    test_left_img, test_right_img, test_left_disp = kitti.dataloader(args.datapath)

    train_loader = paddle.io.DataLoader(dataloader.MyDataloader(train_left_img, train_right_img, train_left_disp, training=True),
                                        batch_size=args.train_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=True, drop_last=False, num_workers=2)
    test_loader = paddle.io.DataLoader(dataloader.MyDataloader(test_left_img, test_right_img, test_left_disp, training=False),
                                       batch_size=args.test_batch_size, places=paddle.CUDAPlace(gpu_id), shuffle=True, drop_last=False, num_workers=2)

    model = Ownnet(args)
    model.train()

    # print(paddle.summary(model, input_size=[(2, 3, 256, 512), (2, 3, 256, 512)]))
    # print(model.parameters())
    # for name, param in model.named_parameters():
    #     print(name, '\n', param)

    if args.resume:
        model_state, _ = fluid.dygraph.load_dygraph("results/"+args.model)
        model.set_dict(model_state)

    batch_len = 0
    for batch_id, data in enumerate(train_loader()):
        batch_len += 1

    print("batch_len", batch_len)

    boundaries = [300 * batch_len, 500 * batch_len]
    values = [args.lr, args.lr*0.1, args.lr*0.01]
    lr_shedule = paddle.optimizer.lr.PiecewiseDecay(boundaries, values, 0)
    temp_lr = 5e-4
    optimizer = paddle.optimizer.Adam(learning_rate=lr_shedule, parameters=model.parameters())

    error_3pixel_check = np.inf

    for epoch in range(args.epoch):
        stage_loss_list = np.zeros((4), dtype=np.float32)
        sum_losses_rec = 0
        error_3pixel_list = np.zeros((4), dtype=np.float32)
        error_3pixel = 0
        lr_shedule.step()

        for batch_id, data in enumerate(train_loader()):

            left_img, right_img, gt = data
            print(left_img.shape)
            outputs = model(left_img, right_img)

            outputs = [paddle.squeeze(output) for output in outputs]

            mask = paddle.to_tensor(gt.numpy() > 0)
            gt_mask = paddle.masked_select(gt, mask)
            print(gt.shape)
            print(outputs[3].shape)

            # loss = args.loss_weights[3] * F.smooth_l1_loss(paddle.masked_select(outputs[3], mask), gt_mask, reduction="mean")
            loss = F.smooth_l1_loss(outputs[3], gt, reduction="mean")

            # print(loss)
            #
            # loss = sum(loss)
            # print(loss)
            #
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()



            # gt_mask = np.zeros(gt.shape, np.float32)
            # gt_mask[gt.numpy() > 0] = 1.0
            # useful_pixels = len(np.flatnonzero(gt_mask))
            # gt_mask = fluid.layers.assign(gt_mask)
            # gt_mask.stop_gradient = True
            # gt = fluid.layers.elementwise_mul(gt, gt_mask)


            # L1_loss = F.smooth_l1_loss()
            tem_stage_loss = []
            # for index in range(stages):
                # print(output[index].squeeze().shape)


                # mask_output = fluid.layers.elementwise_mul(output[index], gt_mask)
                # # temp_loss = fluid.layers.reduce_mean(fluid.layers.smooth_l1(mask_output, gt))
                # temp_loss = F.smooth_l1_loss(, gt)
                # temp_loss = temp_loss * args.loss_weights[index] *(gt.shape[-4]*gt.shape[-3]*gt.shape[-1]*gt.shape[-2]) /useful_pixels
                # tem_stage_loss.append(temp_loss)
                # stage_loss_list[index] += temp_loss.numpy()

            # sum_loss = fluid.layers.sum(tem_stage_loss)
            # sum_loss = tem_stage_loss[3]
            # sum_loss.backward()
            # adam.minimize(sum_loss)

            # sum_losses_rec += sum_loss.numpy()

            info_str = ['Stage {} = {:.2f}'.format(x, float(stage_loss_list[x]/ (batch_id + 1))) for x in range(stages)]
            info_str = '\t'.join(info_str)

            print("Train: epoch {}, batch_id {} learn_lr {:.5f} sum_loss {} \t {}".format(epoch,
                                                                                      batch_id,
                                                                                      optimizer.get_lr(),
                                                                                      np.round(sum_losses_rec / (batch_id + 1),3),
                                                                                      info_str))

            raise SystemExit


        with paddle.no_grad():
            model.eval()

            for batch_id, data in enumerate(test_loader()):
                left_img, right_img, gt = data

                output = model(left_img, right_img)

                for stage in range(stages):
                    error_3pixel_list[stage] += error_estimating(output[stage].numpy(), gt.numpy())
                    error_3pixel = sum(error_3pixel_list)

                info_str = ['Stage {} = {:.2f}'.format(x, float(error_3pixel_list[x] / (batch_id + 1))) for x in range(stages)]
                info_str = '\t'.join(info_str)

                print("Test: epoch {}, batch_id {} error 3pixel {}\t  {}" .format(epoch,
                                                                                  batch_id,
                                                                                  round(error_3pixel/(batch_id+1), 3),
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

                fluid.save_dygraph(model.state_dict(), "results/"+args.model)
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


