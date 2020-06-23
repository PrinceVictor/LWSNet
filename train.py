import argparse
import paddle
import paddle.fluid as fluid

from models.models import *
from dataloader import sceneflow as sf
from dataloader import dataloader

parser = argparse.ArgumentParser(description='finetune KITTI')

parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/home/victor/DATA/sceneflow')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 5, 5])
parser.add_argument('--channels_3d', type=int, default=8, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--resume', action='store_true', default=False,)

args = parser.parse_args()

def network(stages, training = False):

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    # training = True

    net = Ownnet(args)

    if training:
        img_data_shape = [train_batch_size, 3, 256, 512]
        disp_data_shape = [train_batch_size, 1, 256, 512]

    else:
        img_data_shape = [test_batch_size, 3, 256, 512]
        disp_data_shape = [test_batch_size, 1, 256, 512]

    left_image = fluid.data(name='left_img', dtype='float32', shape=img_data_shape)
    right_image = fluid.data(name='right_img', dtype='float32', shape=img_data_shape)
    groundtruth = fluid.data(name='gt', dtype='float32', shape=disp_data_shape)

    data_loader = fluid.io.DataLoader.from_generator(feed_list=[left_image, right_image, groundtruth],
                                                     iterable=True,
                                                     capacity=32)

    output = net.inference(left_image, right_image)
    gt = fluid.layers.clip(groundtruth, min=0.0, max=192.0)

    sum_loss = fluid.layers.zeros(shape=[1], dtype="float32")
    tem_stage_loss = []

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

    stages = 4
    gpu_id = args.gpu_id

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = sf.dataloader(args.datapath)

    # train_left_img, train_right_img, train_left_disp, \
    # test_left_img, test_right_img, test_left_disp = kitti.dataloader(args.datapath)

    train_loader = dataloader.sceneflow_dataloader(train_left_img, train_right_img, train_left_disp, training=True)
    test_loader = dataloader.sceneflow_dataloader(test_left_img, test_right_img, test_left_disp, training=False)

    train_reader = train_loader.create_reader()
    train_batch_reader = paddle.batch(reader=train_reader, batch_size=args.train_batch_size)

    test_reader = test_loader.create_reader()
    test_batch_reader = paddle.batch(reader=test_reader, batch_size=args.test_batch_size)

    train_prog = fluid.default_main_program()
    train_startup = fluid.default_startup_program()

    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)

    with fluid.program_guard(train_prog, train_startup):
        with fluid.unique_name.guard():
            train_ouput, train_stage_loss, train_sum_loss, train_loader = network(stages, training=True)
            adam = fluid.optimizer.Adam(learning_rate=args.lr)
            adam.minimize(train_sum_loss)

    if args.resume:
        fluid.io.load_persistables(executor=exe, dirname="results/model", filename="sceneflow",
                                   main_program=train_prog)

    test_prog = train_prog.clone(for_test=True)

    with fluid.program_guard(test_prog):
        with fluid.unique_name.guard():
            ouput, stage_loss, sum_loss, test_loader = network(stages, training=False)

    exe.run(train_startup)

    test_exe = fluid.Executor(place)

    train_loader.set_sample_list_generator(train_batch_reader, places=fluid.cuda_places(gpu_id))
    test_loader.set_sample_list_generator(test_batch_reader, places=fluid.cuda_places(gpu_id))

    sum_loss_check = np.inf

    for epoch in range(args.epoch):
        stage_loss_list = np.zeros((1, 4), dtype=np.float32)
        sum_losses_rec = 0
        error_3pixel_list = np.zeros((1, 4), dtype=np.float32)
        error_3pixel = 0

        for batch_id, data in enumerate(train_loader()):

            predict, stage_losses, sum_losses = exe.run(program=train_prog,
                                                        feed=data,
                                                        fetch_list=[train_ouput.name,
                                                                    train_stage_loss.name,
                                                                    train_sum_loss.name])

            stage_loss_list += stage_losses
            sum_losses_rec += sum_losses
            print("Train: epoch {}, batch_id {} sum_loss {} stage_loss {}" .format(epoch,
                                                                                   batch_id,
                                                                                   sum_losses_rec/(batch_id+1),
                                                                                   stage_loss_list/(batch_id+1)))


        if sum_losses_rec / (batch_id + 1) < sum_loss_check:
            # fluid.io.save_inference_model(dirname="results/model",
            #                               feeded_var_names=["left_img", "right_img"],
            #                               target_vars=[ouput], executor=exe)
            fluid.io.save_persistables(executor=exe, dirname="results/model", filename="sceneflow", main_program=train_prog)
            print("save model param success")

        for batch_id, data in enumerate(test_loader()):

            predict, stage_losses, sum_losses, gt = test_exe.run(program=test_prog,
                                                             feed=data,
                                                             fetch_list=[ouput.name, stage_loss.name, sum_loss.name, "gt"])

            for stage in range(stages):
                error_3pixel_list[:,stage] += error_estimating(predict[stage], gt)
                error_3pixel = sum(error_3pixel_list)

            print("Test: epoch {}, batch_id {} error 3pixel {} stage_loss {}" .format(epoch,
                                                                                  batch_id,
                                                                                  error_3pixel/(batch_id+1),
                                                                                  error_3pixel_list/(batch_id+1)))

def name_check(var):
    return True
    # if "left_img" in var.name:
    #     return True
    # elif "right_img" in var.name:
    #     return True
    # else:

def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = np.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return float(err3) / float(mask.sum())


if __name__ == "__main__":

    main()


