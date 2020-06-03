import paddle
import paddle.fluid as fluid

from models.submodules import *
from utils.utils import *

class Ownnet():
    def __init__(self):
        self.feature_extraction = feature_extraction()

    def feature_extractor(self, input):
        return self.feature_extraction.inference(input)

    def inference(self, left_input, right_input):

        feats_l = self.feature_extractor(left_input)
        feats_r = self.feature_extractor(right_input)

        return feats_l

        pred = []

        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = fluid.layers.resize_bilinear(pred[scale-1])

            else:
                cost = self

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

        









        # return output