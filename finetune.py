import argparse
import paddle
import paddle.fluid as fluid
import cv2

from paddle.fluid.dygraph.base import to_variable

from models.models import *
from dataloader import kitti2015load as kitti
from dataloader import dataloader

parser = argparse.ArgumentParser(description='finetune KITTI')

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
# parser.add_argument('--datapath', default='/home/liupengchao/zhb/dataset/Kitti/data_scene_flow/training/', help='datapath')
parser.add_argument('--datapath', default='/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/training/', help='datapath')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--resume', action='store_true', default=False,)

args = parser.parse_args()

def main():

    print("finetune KITTI main()")

    stages = 4
    gpu_id = args.gpu_id

    place = fluid.CUDAPlace(gpu_id)
    fluid.enable_imperative(place)

    train_left_img, train_right_img, train_left_disp, \
    test_left_img, test_right_img, test_left_disp = kitti.dataloader(args.datapath)

    train_loader = dataloader.ImageLoad(train_left_img, train_right_img, train_left_disp, training=True)
    test_loader = dataloader.ImageLoad(test_left_img, test_right_img, test_left_disp, training=False)

    train_reader = train_loader.create_reader()
    train_batch_reader = paddle.batch(reader=train_reader, batch_size=args.train_batch_size)

    test_reader = test_loader.create_reader()
    test_batch_reader = paddle.batch(reader=test_reader, batch_size=args.test_batch_size)

    train_data_loader = fluid.io.DataLoader.from_generator(iterable=True, capacity=4)
    train_data_loader.set_sample_list_generator(train_batch_reader, places=fluid.cuda_places(gpu_id))

    test_data_loader = fluid.io.DataLoader.from_generator(iterable=True, capacity=4)
    test_data_loader.set_sample_list_generator(test_batch_reader, places=fluid.cuda_places(gpu_id))

    model = Ownnet(args)

    if args.resume:
        model_state, _ = fluid.dygraph.load_dygraph("results/kitti")
        model.set_dict(model_state)

    adam = fluid.optimizer.AdamOptimizer(learning_rate=args.lr, parameter_list=model.parameters())

    error_3pixel_check = np.inf

    for epoch in range(args.epoch):
        stage_loss_list = np.zeros((4), dtype=np.float32)
        sum_losses_rec = 0
        error_3pixel_list = np.zeros((4), dtype=np.float32)
        error_3pixel = 0

        model.train()

        for batch_id, data in enumerate(train_data_loader()):
            left_img, right_img, gt = data

            gt_mask = np.zeros(gt.shape, np.float32)
            gt_mask[gt.numpy() > 0] = 1.0
            gt_mask = fluid.layers.assign(gt_mask)

            output = model(left_img, right_img)


            tem_stage_loss = []
            for index in range(stages):

                mask_output = fluid.layers.elementwise_mul(output[index], gt_mask)
                temp_loss = fluid.layers.reduce_mean(fluid.layers.smooth_l1(mask_output, gt)/(gt.shape[-1]*gt.shape[-2]))
                temp_loss = temp_loss * args.loss_weights[index]
                tem_stage_loss.append(temp_loss)
                stage_loss_list[index] += temp_loss.numpy()

            sum_loss = fluid.layers.sum(tem_stage_loss)
            sum_loss.backward()
            adam.minimize(sum_loss)
            model.clear_gradients()
            # raise StopIteration

            sum_losses_rec += sum_loss.numpy()

            info_str = ['Stage {} = {:.2f}'.format(x, float(stage_loss_list[x]/ (batch_id + 1))) for x in range(stages)]
            info_str = '\t'.join(info_str)

            print("Train: epoch {}, batch_id {} sum_loss {} \t {}".format(epoch,
                                                                          batch_id,
                                                                          np.round(sum_losses_rec / (batch_id + 1),3),
                                                                          info_str))

        model.eval()

        for batch_id, data in enumerate(test_data_loader()):
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

            for batch_size_I in range(args.test_batch_size):
                disp = (output[3][batch_size_I][0].numpy()*256).astype(np.uint8)
                disp = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=1.2, beta=0), cv2.COLORMAP_JET)

                gt_disp = (gt[batch_size_I][0].numpy()*256).astype(np.uint8)
                gt_disp = cv2.applyColorMap(cv2.convertScaleAbs(gt_disp, alpha=1.2, beta=0), cv2.COLORMAP_JET)
                cv2.imshow("disp", disp)
                cv2.imshow("gt_disp", gt_disp)
                cv2.waitKey(0)

        if error_3pixel / (batch_id + 1) < error_3pixel_check:
            error_3pixel_check = error_3pixel / (batch_id + 1)

            fluid.save_dygraph(model.state_dict(), "results/kitti")
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


