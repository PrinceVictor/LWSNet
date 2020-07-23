import paddle
import paddle.fluid as fluid

from models.submodules import *
from utils.utils import *

class Ownnet(fluid.dygraph.Layer):
    def __init__(self, args):
        super(Ownnet, self).__init__()

        self.maxdisplist = args.maxdisplist
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate

        self.feature_extraction = feature_extraction()
        self.volume_postprocess = []

        for i in range(3):
            net3d = Post_3DConvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)

        self.refinement1_left = refinement1(channels=32)
        self.refinement1_disp = refinement1(channels=32)
        self.refinement2 = refinement2(channels=32)


    def feature_extractor(self, input):
        return self.feature_extraction.inference(input)

    def warp(self, x, disp):
        """

        :param x:[B, C, H, W] flo:[B, 2, H, W] flow
        :param disp:
        :return:
        """
        B, C, H, W = x.shape

        xx = fluid.layers.expand(
            fluid.layers.reshape(fluid.layers.range(0, W, 1, dtype='float32'), shape=[1, -1]),
            expand_times=[H, 1])

        yy = fluid.layers.expand(
            fluid.layers.reshape(fluid.layers.range(0, H, 1, dtype='float32'), shape=[-1, 1]),
            expand_times=[1, W])

        xx = fluid.layers.expand(fluid.layers.reshape(xx, shape=[1, 1, H, W]),
                                 expand_times=[B, 1, 1, 1])

        yy = fluid.layers.expand(fluid.layers.reshape(yy, shape=[1, 1, H, W]),
                                 expand_times=[B, 1, 1, 1])

        # vgrid = fluid.layers.concat([xx, yy], 1)

        xx_disp_sub = fluid.layers.elementwise_sub(xx, disp)

        vgrid = fluid.layers.concat([xx_disp_sub, yy], 1)


        # fluid.layers.assign(fluid.layers.elementwise_sub(vgrid[:, :1, :, :], disp), vgrid[:, :1, :, :])

        # vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

        # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = fluid.layers.transpose(vgrid, [0, 2, 3, 1])
        output = fluid.layers.grid_sampler(x, vgrid)

        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0

        cost_list = []
        for i in range(0, maxdisp, stride):

            if i > 0:

                cost_left = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :i]), dim=1, keep_dim=True)
                cost_right = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i]),
                                                     dim=1, keep_dim=True)

                temp_cost = fluid.layers.concat(input=[cost_left, cost_right], axis=-1)
            else:
                temp_cost = fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :] - feat_r[:, :, :, :]),
                                                    dim=1, keep_dim=True)

            cost_list.append(temp_cost)

        cost = fluid.layers.concat(cost_list, axis=1)

        return cost

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):

        shape = feat_l.shape

        # batch_disp = fluid.layers.expand(unsqueeze(disp, [1]), [1, maxdisp*2-1, 1, 1, 1])
        # batch_disp = fluid.layers.reshape(batch_disp, shape=[-1, 1, shape[-2], shape[-1]], inplace=True)

        batch_disp = unsqueeze_repeat_view(disp, maxdisp, [-1, 1, shape[-2], shape[-1]])

        batch_shfit = fluid.layers.expand(fluid.layers.range(-maxdisp+1, maxdisp, step=1, dtype='int32'), [shape[0]])
        batch_shfit = fluid.layers.reshape(batch_shfit, [-1,1,1,1]) * stride

        batch_disp = batch_disp - fluid.layers.cast(batch_shfit, 'float32')

        batch_feat_l = unsqueeze_repeat_view(feat_l, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])
        batch_feat_r = unsqueeze_repeat_view(feat_r, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])

        cost = fluid.layers.reduce_sum(
            fluid.layers.abs(batch_feat_l - self.warp(batch_feat_r, batch_disp)),
            dim=1)

        cost = fluid.layers.reshape(cost, shape=[shape[0], -1, shape[2], shape[3]])

        return cost

    def foward(self, left_input, right_input):

        img_size = left_input.shape

        feats_l = self.feature_extractor(left_input)
        feats_r = self.feature_extractor(right_input)

        # return feats_l

        pred = []



        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = fluid.layers.resize_bilinear(pred[scale - 1],
                                                     out_shape=[feats_l[scale].shape[2], feats_l[scale].shape[3]]) * feats_l[scale].shape[2] / img_size[2]

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

            cost = unsqueeze(cost, [1])
            cost = self.volume_postprocess[scale].post_3dconvs(cost) + cost
            cost = fluid.layers.squeeze(cost, [1])

            if scale == 0:
                pre_low_res = disparity_regression(input=fluid.layers.softmax(-cost, axis=1),
                                                   start=0,
                                                   end=self.maxdisplist[0])
                pre_low_res = pre_low_res * img_size[2]/pre_low_res.shape[2]

                disp_up = fluid.layers.resize_bilinear(pre_low_res,
                                                       out_shape=[img_size[2], img_size[3]])
                pred.append(disp_up)
            else:
                pre_low_res = disparity_regression(input=fluid.layers.softmax(-cost, axis=1),
                                                   start=-self.maxdisplist[scale]+1,
                                                   end=self.maxdisplist[scale])
                pre_low_res = pre_low_res * img_size[2] / pre_low_res.shape[2]

                disp_up = fluid.layers.resize_bilinear(pre_low_res,
                                                       out_shape=[img_size[2], img_size[3]])
                pred.append(disp_up+pred[scale-1])

        refined_left = self.refinement1_left.inference(left_input)
        refined_disp = self.refinement1_disp.inference(pred[-1])
        disp = self.refinement2.inference(input=fluid.layers.concat([refined_left, refined_disp], 1))
        disp_up = fluid.layers.resize_bilinear(input=disp,
                                               out_shape=[img_size[2], img_size[3]])
        pred.append(pred[2]+disp_up)

        return pred


def disparity_regression(input, start, end, stride=1):
    disp = fluid.layers.range(start*stride, end*stride, stride, dtype='float32')

    disp.stop_gradient = True
    disp = fluid.layers.reshape(disp, shape=[1, -1, 1, 1])

    disp = fluid.layers.expand(disp,
                               expand_times=[input.shape[0], 1, input.shape[2], input.shape[3]])
    # disp = fluid.layers.expand(disp,
    #                            expand_times=[1, 1, input.shape[2], input.shape[3]])

    output = fluid.layers.reduce_sum(input*disp, dim=1, keep_dim=True)
    return output