import paddle
import paddle.fluid as fluid

from models.submodules import *
from utils.utils import *

class Ownnet():
    def __init__(self):
        self.feature_extraction = feature_extraction()

        self.volume_postprocess = []

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

        vgrid = fluid.layers.concat([xx, yy], 1)

        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

        # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = fluid.layers.transpose(vgrid, [0, 2, 3, 1])
        output = fluid.layers.grid_sampler(x, vgrid)

        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0

        cost_shape = [feat_l.shape[0], maxdisp//stride, feat_l.shape[2], feat_l.shape[3]]
        cost = fluid.data(name='cost', shape=cost_shape, dtype='float32')

        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, : i] = \
                fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :i]), dim=1)

            if i > 0:
                cost[:, i // stride, :, i:] = \
                    fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :i] - feat_r[:, :, :, :-i]), dim=1)

            else:
                cost[:, i // stride, :, i:] = \
                    fluid.layers.reduce_sum(fluid.layers.abs(feat_l[:, :, :, :] - feat_r[:, :, :, :]), dim=1)

        return cost

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):

        shape = feat_l.shape

        # batch_disp = fluid.layers.expand(unsqueeze(disp, [1]), [1, maxdisp*2-1, 1, 1, 1])
        # batch_disp = fluid.layers.reshape(batch_disp, shape=[-1, 1, shape[-2], shape[-1]], inplace=True)

        batch_disp = unsqueeze_repeat_view(disp, maxdisp, [-1, 1, shape[-2], shape[-1]])

        batch_shfit = fluid.layers.expand(fluid.layers.range(-maxdisp+1, maxdisp, dtype='int32'), shape[0])
        batch_shfit = fluid.layers.reshape(batch_shfit, [-1,1,1,1]) * stride

        batch_disp = batch_disp - fluid.layers.cast(batch_shfit, 'float32')

        batch_feat_l = unsqueeze_repeat_view(feat_l, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])
        batch_feat_r = unsqueeze_repeat_view(feat_r, maxdisp, [-1, shape[-3], shape[-2], shape[-1]])

        cost = fluid.layers.reduce_sum(
            fluid.layers.abs(batch_feat_l - self.warp(batch_feat_r, batch_disp)),
            dim=1)

        cost = fluid.layers.reshape(cost, shape=[shape[0], -1, shape[2], shape[3]])

        return cost

    def inference(self, left_input, right_input):

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
