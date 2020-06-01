import paddle
import paddle.fluid as fluid

from models.submodules import *

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
            cost[:, i//stride, :, : i] = fluid.layers.abs(feat_l[:, :, :, :i])




        # return output