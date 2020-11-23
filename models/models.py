import paddle
import paddle.nn
import paddle.nn.functional as F

from models.submodules import *
from utils.utils import *

import torch

class Ownnet(nn.Layer):
    def __init__(self, args):
        super(Ownnet, self).__init__()

        self.maxdisplist = args.maxdisplist
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate

        self.feature_extraction = feature_extraction()
        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.LayerList(self.volume_postprocess)

        self.refinement1_left = refinement1(in_channels=3, out_channels=32)
        self.refinement1_disp = refinement1(in_channels=1, out_channels=32)
        self.refinement2 = refinement2(in_channels=64, out_channels=32)

    def warp(self, x, disp):
        """

        :param x:[B, C, H, W] flo:[B, 2, H, W] flow
        :param disp:
        :return:
        """
        B, C, H, W = x.shape

        xx = paddle.expand(paddle.arange(0, W, step=1, dtype='float32').reshape(shape=[1, -1]), shape=[H, W])
        yy = paddle.expand(paddle.arange(0, H, step=1, dtype='float32').reshape(shape=[-1, 1]), shape=[H, W])

        # xx = fluid.layers.expand(
        #     fluid.layers.reshape(fluid.layers.range(0, W, 1, dtype='float32'), shape=[1, -1]),
        #     expand_times=[H, 1])
        # yy = fluid.layers.expand(
        #     fluid.layers.reshape(fluid.layers.range(0, H, 1, dtype='float32'), shape=[-1, 1]),
        #     expand_times=[1, W])

        xx = paddle.expand(xx.reshape(shape=[1, 1, H, W]), shape=[B, 1, H, W])
        yy = paddle.expand(yy.reshape(shape=[1, 1, H, W]), shape=[B, 1, H, W])

        # xx = fluid.layers.expand(fluid.layers.reshape(xx, shape=[1, 1, H, W]),
        #                          expand_times=[B, 1, 1, 1])
        # yy = fluid.layers.expand(fluid.layers.reshape(yy, shape=[1, 1, H, W]),
        #                          expand_times=[B, 1, 1, 1])

        # vgrid = fluid.layers.concat([xx, yy], 1)
        vgrid = paddle.concat((xx, yy), axis=1)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:]-disp

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        # xx_disp_sub = fluid.layers.elementwise_sub(xx, disp)
        # xx_disp_sub = xx_disp_sub/max(W-1, 1)*2.0 - 1.0
        # yy = yy/max(H-1, 1)*2.0 - 1.0

        # vgrid = fluid.layers.concat([xx_disp_sub, yy], 1)


        # fluid.layers.assign(fluid.layers.elementwise_sub(vgrid[:, :1, :, :], disp), vgrid[:, :1, :, :])

        # vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

        # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = paddle.transpose(vgrid, [0, 2, 3, 1])
        output = F.grid_sample(x, vgrid)

        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0

        cost = paddle.zeros((feat_l.shape[0], maxdisp//stride, feat_l.shape[2], feat_l.shape[3]), dtype='float32')
        # cost.stop_gradient = False

        # cost_list = []
        for i in range(0, maxdisp, stride):

            if i > 0:
                cost[:, i // stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
                cost[:, i//stride, :, i:] = paddle.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i//stride, :, i:] = paddle.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

            # if i > 0:
            #     cost_left = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :i]), dim=1, keep_dim=True)
            #     cost_right = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i]),
            #                                          dim=1, keep_dim=True)
            #
            #     temp_cost = fluid.layers.concat(input=[cost_left, cost_right], axis=-1)
            # else:
            #     temp_cost = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :] - feat_r[:, :, :, :]),
            #                                         dim=1, keep_dim=True)
            #
            # cost_list.append(temp_cost)

        # cost = fluid.layers.concat(cost_list, axis=1)

        return cost

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):

        size = feat_l.shape

        # batch_disp = fluid.layers.expand(unsqueeze(disp, [1]), [1, maxdisp*2-1, 1, 1, 1])
        # batch_disp = fluid.layers.reshape(batch_disp, shape=[-1, 1, shape[-2], shape[-1]], inplace=True)
        disp = paddle.unsqueeze(disp, axis=1)
        batch_disp = paddle.expand(disp, shape=[disp.shape[0], maxdisp * 2 - 1, disp.shape[-3], disp.shape[-2], disp.shape[-1]])
        batch_disp = batch_disp.reshape(shape=[-1, 1, size[-2], size[-1]])

        batch_shift = paddle.arange(-maxdisp + 1, maxdisp, dtype="float32")
        batch_shift = paddle.expand(batch_shift, shape=[size[0], batch_shift.shape[0]]).reshape(shape=[-1]).unsqueeze(axis=[1,2,3]) * stride
        batch_disp = batch_disp - batch_shift
        batch_feat_l = paddle.unsqueeze(feat_l, axis=1).expand(shape=[size[0], maxdisp * 2 - 1, size[-3], size[-2], size[-1]]).reshape(shape=[-1, size[-3], size[-2], size[-1]])
        batch_feat_r = paddle.unsqueeze(feat_r, axis=1).expand(shape=[size[0], maxdisp * 2 - 1, size[-3], size[-2], size[-1]]).reshape(shape=[-1, size[-3], size[-2], size[-1]])
        # batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).reshape(-1, size[-3], size[-2], size[-1])
        # batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).reshape(-1, size[-3], size[-2], size[-1])
        cost = paddle.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.reshape(shape=[size[0], -1, size[2], size[3]])

        # batch_disp = unsqueeze_repeat_view(disp, maxdisp, [-1, 1, shape[-2], shape[-1]])
        #
        # batch_shift_temp = fluid.layers.expand(fluid.layers.range(-maxdisp+1, maxdisp, step=1, dtype='float32'), [shape[0]])
        # batch_shfit = fluid.layers.reshape(batch_shift_temp, shape=[-1,1,1,1])
        #
        # batch_disp = batch_disp - batch_shfit * stride
        #
        # batch_feat_l = unsqueeze_repeat_view(feat_l, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])
        # batch_feat_r = unsqueeze_repeat_view(feat_r, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])
        #
        # cost = fluid.layers.reduce_sum(
        #     fluid.layers.abs(batch_feat_l - self.warp(batch_feat_r, batch_disp)),
        #     dim=1, keep_dim=True)
        #
        # cost = fluid.layers.reshape(cost, shape=[shape[0], -1, shape[2], shape[3]])

        return cost

    def forward(self, left_input, right_input):

        img_size = left_input.shape

        feats_l = self.feature_extraction(left_input)
        feats_r = self.feature_extraction(right_input)

        # return feats_l

        pred = []

        for scale in range(len(feats_l)):

            if scale > 0:
                wflow = F.interpolate(pred[scale - 1], size=[feats_l[scale].shape[2], feats_l[scale].shape[3]], mode="bilinear") *\
                        feats_l[scale].shape[2] / img_size[2]

                cost = self._build_volume_2d3(feats_l[scale],
                                              feats_r[scale],
                                              self.maxdisplist[scale],
                                              wflow,
                                              stride=1)

            else:
                cost = self._build_volume_2d(feats_l[scale],
                                             feats_r[scale],
                                             self.maxdisplist[scale],
                                             stride=1)

            cost = paddle.unsqueeze(cost, [1])
            cost = self.volume_postprocess[scale](cost) + cost
            cost = paddle.squeeze(cost, [1])

            if scale == 0:

                pre_low_res = disparity_regression(start=0, end=self.maxdisplist[0])(input=F.softmax(-cost, axis=1))
                pre_low_res = pre_low_res * img_size[2]/pre_low_res.shape[2]
                disp_up = F.interpolate(pre_low_res, size=[img_size[2], img_size[3]], mode="bilinear")

                pred.append(disp_up)
            else:

                pre_low_res = disparity_regression(start=-self.maxdisplist[scale]+1,
                                                   end=self.maxdisplist[scale])(input=F.softmax(-cost, axis=1))
                pre_low_res = pre_low_res * img_size[2] / pre_low_res.shape[2]
                disp_up = F.interpolate(pre_low_res, size=[img_size[2], img_size[3]], mode="bilinear")

                pred.append(disp_up+pred[scale-1])

        refined_left = self.refinement1_left(left_input)
        refined_disp = self.refinement1_disp(pred[-1])
        disp = self.refinement2(input=paddle.concat([refined_left, refined_disp], 1))
        disp_up = F.interpolate(disp, size=[img_size[2], img_size[3]], mode="bilinear")
        pred.append(pred[2]+disp_up)

        return pred


class disparity_regression(nn.Layer):
    def __init__(self, start, end, stride=1):
        super(disparity_regression, self).__init__()
        self.disp = paddle.arange(start*stride, end*stride, stride, dtype='float32')
        # self.disp.stop_gradient = True
        self.disp = paddle.reshape(self.disp, shape=[1, -1, 1, 1])
        _, self.my_steplength, _, _ = self.disp.shape

    def forward(self, input):

        disp = paddle.expand(self.disp, (input.shape[0], self.my_steplength, input.shape[2], input.shape[3]))
        output = paddle.sum(input*disp, axis=1, keepdim=True)
        return output