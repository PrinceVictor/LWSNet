import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.submodules import feature_extraction, post_3dconvs, refinement1, refinement2

class LWSNet(nn.Layer):
    def __init__(self, args):
        super(LWSNet, self).__init__()

        self.maxdisplist = args.maxdisplist
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate

        self.feature_extraction = feature_extraction()
        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.LayerList(self.volume_postprocess) #3D CNN in Stage 1 to Stage 3

        self.refinement1_left = refinement1(in_channels=3, out_channels=32) #input: left image output: left features
        self.refinement1_disp = refinement1(in_channels=1, out_channels=32) #input: disparity stage 3 output: disparity features
        self.refinement2 = refinement2(in_channels=64, out_channels=32)

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        disp: [B, 1, H, W]
        flo: [B, 2, H, W] flow
        output: [B, C, H, W] (im1)
        """
        B, C, H, W = x.shape
        # mesh grid
        xx = paddle.expand(paddle.arange(0, W, step=1, dtype='float32').reshape(shape=[1, -1]), shape=[H, W])
        yy = paddle.expand(paddle.arange(0, H, step=1, dtype='float32').reshape(shape=[-1, 1]), shape=[H, W])

        xx = paddle.expand(xx.reshape(shape=[1, 1, H, W]), shape=[B, 1, H, W])
        yy = paddle.expand(yy.reshape(shape=[1, 1, H, W]), shape=[B, 1, H, W])

        vgrid = paddle.concat((xx, yy), axis=1) #[B, 2, H, W]
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = paddle.transpose(vgrid, [0, 2, 3, 1]) #[B, H, W, 2]
        vgrid.stop_gradient = False

        output = F.grid_sample(x, vgrid)

        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        """
        output full disparity map
        L1 distance-based cost
        """
        assert maxdisp % stride == 0

        cost = paddle.zeros((feat_l.shape[0], maxdisp // stride, feat_l.shape[2], feat_l.shape[3]), dtype='float32')
        cost.stop_gradient=False

        for i in range(0, maxdisp, stride):

            if i > 0:
                cost[:, i // stride, :, :i] = feat_l[:, :, :, :i].abs().sum(axis=1) #occlusion regions
                cost[:, i // stride, :, i:] = paddle.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i // stride, :, i:] = paddle.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        """
        output residual map
        L1 distance-based cost
        """
        size = feat_l.shape

        disp = paddle.unsqueeze(disp, axis=1)
        batch_disp = paddle.expand(disp, shape=[disp.shape[0], maxdisp * 2 - 1, disp.shape[-3], disp.shape[-2],
                                                disp.shape[-1]])
        batch_disp = batch_disp.reshape(shape=[-1, 1, size[-2], size[-1]])

        batch_shift = paddle.arange(-maxdisp + 1, maxdisp, dtype="float32")
        batch_shift = paddle.expand(batch_shift, shape=[size[0], batch_shift.shape[0]]).reshape(shape=[-1]).unsqueeze(
            axis=[1, 2, 3]) * stride
        batch_disp = batch_disp - batch_shift
        batch_feat_l = paddle.unsqueeze(feat_l, axis=1).expand(
            shape=[size[0], maxdisp * 2 - 1, size[-3], size[-2], size[-1]]).reshape(
            shape=[-1, size[-3], size[-2], size[-1]])
        batch_feat_r = paddle.unsqueeze(feat_r, axis=1).expand(
            shape=[size[0], maxdisp * 2 - 1, size[-3], size[-2], size[-1]]).reshape(
            shape=[-1, size[-3], size[-2], size[-1]])

        cost = paddle.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1) #output residual map
        cost = cost.reshape(shape=[size[0], -1, size[2], size[3]])

        return cost

    def forward(self, left_input, right_input):

        img_size = left_input.shape

        feats_l = self.feature_extraction(left_input) #left features
        feats_r = self.feature_extraction(right_input) #right features

        pred = []

        for scale in range(len(feats_l)):

            if scale > 0:
                #stage 2 and stage 3
                wflow = F.interpolate(pred[scale - 1], size=[feats_l[scale].shape[2], feats_l[scale].shape[3]],
                                      mode="bilinear") * \
                        feats_l[scale].shape[2] / img_size[2] #resize disparity of last stage to current resolution

                cost = self._build_volume_2d3(feats_l[scale],
                                              feats_r[scale],
                                              self.maxdisplist[scale],
                                              wflow,
                                              stride=1) #build cost volume

            else:
                #stage 1
                cost = self._build_volume_2d(feats_l[scale],
                                             feats_r[scale],
                                             self.maxdisplist[scale],
                                             stride=1) #build cost volume

            cost = paddle.unsqueeze(cost, [1])
            cost = self.volume_postprocess[scale](cost) + cost #3D CNN, skip connection
            cost = paddle.squeeze(cost, [1])

            if scale == 0:
                #stage 1
                pre_low_res = disparity_regression(start=0, end=self.maxdisplist[0])(input=F.softmax(-cost, axis=1)) #full disparity
                #softmax function computes the probability of a pixel's disparity to be d
                #'softmax(cost)' or 'softmax(-cost)' do not affect the performance because feature-based cost volume provided flexibility.
                pre_low_res = pre_low_res * img_size[2] / pre_low_res.shape[2] #transform disparity value to original resolution
                disp_up = F.interpolate(pre_low_res, size=[img_size[2], img_size[3]], mode="bilinear") #upsample to original resolution

                pred.append(disp_up)
            else:
                #stage 2 and 3
                pre_low_res = disparity_regression(start=-self.maxdisplist[scale] + 1,
                                                   end=self.maxdisplist[scale])(input=F.softmax(-cost, axis=1)) #residual
                pre_low_res = pre_low_res * img_size[2] / pre_low_res.shape[2] #transform residual value to original resolution
                disp_up = F.interpolate(pre_low_res, size=[img_size[2], img_size[3]], mode="bilinear") #upsample to original resolution

                pred.append(disp_up + pred[scale - 1]) #skip connection

        refined_left = self.refinement1_left(left_input)
        refined_disp = self.refinement1_disp(pred[-1])
        disp = self.refinement2(input=paddle.concat([refined_left, refined_disp], 1))
        disp_up = F.interpolate(disp, size=[img_size[2], img_size[3]], mode="bilinear")
        pred.append(pred[2] + disp_up) #skip connection

        return pred #disparity maps of 4 stages


class disparity_regression(nn.Layer):
    def __init__(self, start, end, stride=1):
        super(disparity_regression, self).__init__()
        self.disp = paddle.arange(start * stride, end * stride, stride, dtype='float32')
        self.disp.stop_gradient = True
        self.disp = paddle.reshape(self.disp, shape=[1, -1, 1, 1])
        _, self.my_steplength, _, _ = self.disp.shape

    def forward(self, input):

        disp = paddle.expand(self.disp, (input.shape[0], self.my_steplength, input.shape[2], input.shape[3]))
        output = paddle.sum(input * disp, axis=1, keepdim=True) #compute expectation
        return output