import argparse
import paddle
import paddle.fluid as fluid

from models.models import *
from dataloader import kitti2015load as kitti
from dataloader import dataloader

parser = argparse.ArgumentParser(description='finetune KITTI')

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/home/victor/DATA/kitti_dataset/scene_flow/data_scene_flow/training/',
help='datapath')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
args = parser.parse_args()

def network(net, training = False, test_batch_size=1, train_batch_size=1):

    stages = 4

    if training:
        img_data_shape = [train_batch_size, 3, 256, 512]
        disp_data_shape = [train_batch_size, 1, 256, 512]

    else:
        img_data_shape = [test_batch_size, 3, 368, 1232]
        disp_data_shape = [test_batch_size, 1, 368, 1232]

    left_image = fluid.data(name='left_img', dtype='float32', shape=img_data_shape)
    right_image = fluid.data(name='right_img', dtype='float32', shape=img_data_shape)
    stage_loss = fluid.data(name="stage_loss", dtype="float32", shape=[stages])

    groundtruth = fluid.data(name='gt', dtype='float32', shape=disp_data_shape)

    data_loader = fluid.io.DataLoader.from_generator(feed_list=[left_image, right_image, groundtruth],
                                                     iterable=True,
                                                     capacity=10)

    output = net.inference(left_image, right_image)
    gt = fluid.layers.clip(groundtruth, min=0.0, max=192.0)

    sum_loss = fluid.layers.zeros(shape=[1], dtype="float32")
    tem_stage_loss = []

    if training:
        for index in range(stages):
            temp_predict = fluid.layers.clip(output[index], min=0.0, max=192.0)
            temp_loss = fluid.layers.reduce_mean(fluid.layers.smooth_l1(temp_predict, gt),
                                                 dim=0)
            temp_loss = temp_loss * args.loss_weights[index]
            tem_stage_loss.append(temp_loss)
            sum_loss += temp_loss

    # stage_loss = fluid.layers.stack([stage_loss[0], stage_loss[1], stage_loss[2], stage_loss[3]], axis=-1)
    stage_loss = fluid.layers.concat(input=tem_stage_loss, axis=0)
    predict_ouput = fluid.layers.concat(input=output, axis=0)

    return predict_ouput, stage_loss, sum_loss, data_loader

def main():

    print("finetune KITTI main()")

    train_left_img, train_right_img, train_left_disp, \
    test_left_img, test_right_img, test_left_disp = kitti.dataloader(args.datapath)

    train_loader = dataloader.ImageLoad(train_left_img, train_right_img, train_left_disp, training=True)
    test_loader = dataloader.ImageLoad(test_left_img, test_right_img, test_left_disp, training=False)

    trian_reader = train_loader.create_reader()
    batch_reader = paddle.batch(reader=trian_reader, batch_size=5)

    net = Ownnet(args)

    train_prog = fluid.Program()
    train_startup = fluid.Program()

    with fluid.program_guard(train_prog, train_startup):
        with fluid.unique_name.guard():
            ouput, stage_loss, sum_loss, train_loader = network(net, training=True, train_batch_size=5)
            adam = fluid.optimizer.Adam(learning_rate=args.lr)
            adam.minimize(sum_loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(train_startup)

    train_loader.set_sample_list_generator(batch_reader, places=fluid.cuda_places(0))

    for data in train_loader():
        predict, stage_losses, sum_losses = exe.run(program=train_prog,
                                                    feed=data,
                                                    fetch_list=[ouput.name, stage_loss.name, sum_loss.name])
        print(sum_losses)
        print(stage_losses)
        print(predict.shape)

if __name__ == "__main__":

    main()


