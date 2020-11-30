import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def convbn(in_channels, out_channels,
           kernel_size, stride, padding, dilation=1,
           conv_param_attr=None, conv_bias_attr=None,
           bn_param_attr=None, bn_bias_attr=None):
    # 2D convolutional layer + batchnorm
    return nn.Sequential(nn.Conv2D(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=dilation if dilation>1 else padding,
                                   dilation=dilation,
                                   weight_attr=conv_param_attr,
                                   bias_attr=conv_bias_attr),
                         nn.BatchNorm2D(num_features=out_channels))

def deconvbn(in_channels, out_channels,
             kernel_size, stride, padding, output_padding=1, dilation=1,
             conv_param_attr=None, conv_bias_attr=None,
             bn_param_attr=None, bn_bias_attr=None):
    # 2D deconvolutional layer + batchnorm
    return nn.Sequential(nn.Conv2DTranspose(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            output_padding=output_padding,
                                            stride=stride,
                                            weight_attr=conv_param_attr,
                                            bias_attr=conv_bias_attr),
                         nn.BatchNorm2D(num_features=out_channels))

class hourglass(nn.Layer):
    def __init__(self, init_channel=8):
        super(hourglass, self).__init__()
        self.init_channel = init_channel

        self.conv1 = nn.Sequential(convbn(in_channels=self.init_channel,
                                          out_channels=self.init_channel*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(convbn(in_channels=self.init_channel * 2,
                                          out_channels=self.init_channel * 2,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(convbn(in_channels=self.init_channel * 2,
                                          out_channels=self.init_channel * 2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(convbn(in_channels=self.init_channel * 2,
                                          out_channels=self.init_channel * 2,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU())

        self.conv5 = deconvbn(in_channels=self.init_channel * 2,
                              out_channels=self.init_channel * 2,
                              kernel_size=3,
                              padding=1,
                              output_padding=1,
                              stride=2,
                              conv_param_attr=nn.initializer.KaimingNormal(),
                              conv_bias_attr=False) #+conv2

        self.conv6 = deconvbn(in_channels=self.init_channel * 2,
                              out_channels=self.init_channel,
                              kernel_size=3,
                              padding=1,
                              output_padding=1,
                              stride=2,
                              conv_param_attr=nn.initializer.KaimingNormal(),
                              conv_bias_attr=False)

    def forward(self, input):
        res = []
        output = self.conv1(input) #in: 1/2 out: 1/4 channel: 8 to 16
        pre = self.conv2(output) #in: 1/4 out: 1/4 channel: 16 to 16

        output = self.conv3(pre) #in: 1/4 out: 1/8 channel: 16 to 16
        output = self.conv4(output) #in: 1/8 out: 1/8 channel: 16 to 16
        res.append(output) #feature maps(1/8) channel: 16

        post = F.relu(self.conv5(output)+pre) #in: 1/8 out: 1/4 channel: 16 to 16
        res.append(post) #feature maps(1/4) channel: 16

        output = self.conv6(post) #in: 1/4 out: 1/2 channel: 16 to 16
        res.append(output) #feature maps(1/2) channel: 8

        return res #feature maps(1/8, 1/4, 1/2)



class feature_extraction(nn.Layer):

    def __init__(self):
        super(feature_extraction, self).__init__()

        self.dres0 = nn.Sequential(convbn(in_channels=3,
                                          out_channels=4,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          dilation=2,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU(),
                                   convbn(in_channels=4,
                                          out_channels=8,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          dilation=4,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU())

        self.dres1 = nn.Sequential(convbn(in_channels=8,
                                          out_channels=4,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          dilation=2,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False),
                                   nn.ReLU(),
                                   convbn(in_channels=4,
                                          out_channels=8,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          dilation=2,
                                          conv_param_attr=nn.initializer.KaimingNormal(),
                                          conv_bias_attr=False))

        self.dres2 = hourglass(init_channel=8)

        self.classif1 = nn.Sequential(convbn(in_channels=8,
                                             out_channels=8,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             dilation=1,
                                             conv_param_attr=nn.initializer.KaimingNormal(),
                                             conv_bias_attr=False),
                                      nn.ReLU(),
                                      nn.Conv2D(in_channels=8,
                                                out_channels=8,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1,
                                                weight_attr=nn.initializer.KaimingNormal(),
                                                bias_attr=False))



    def forward(self, input):

        output = self.dres0(input) #in: 1 out: 1/2 channel: 3 to 8
        output = self.dres1(output) + output #in: 1/2 out: 1/2 channel: 8 to 8

        res = self.dres2(output) #in: 1/2 out: 1/8, 1/4, 1/2
        output = res[-1] + output #skip connection

        output = self.classif1(output) #in: 1/2 out: 1/2 channel: 8 to 8
        res.pop(-1)
        res.append(output) #feature maps(1/2) channel: 8

        return res

def batch_relu_conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bn3d=True,
                      conv_param_attr=nn.initializer.KaimingNormal(), conv_bias_attr=False,
                      bn_param_attr=None, bn_bias_attr=None):
    if bn3d:
        # 3D batchnorm + relu + convolutional layer
        return nn.Sequential(nn.BatchNorm3D(num_features=in_channels),
                             nn.ReLU(),
                             nn.Conv3D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       weight_attr=conv_param_attr,
                                       bias_attr=conv_bias_attr))
    else:
        # 3D relu + convolutional layer
        return nn.Sequential(nn.ReLU(),
                             nn.Conv3D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       weight_attr=conv_param_attr,
                                       bias_attr=conv_bias_attr))

def post_3dconvs(layers, channels):
    #3D CNN applied on cost volume
    net = [batch_relu_conv3d(1, channels)]
    net = net+[batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net = net+[batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)

def preconv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        # 2D batchnorm + relu + convolutional layer
        return nn.Sequential(nn.BatchNorm2D(num_features=in_channels),
                             nn.ReLU(),
                             nn.Conv2D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=dilation if dilation > 1 else pad,
                                       dilation=dilation,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=False))


def preconv2d_depthseperated(in_channels, out_channels,
                             kernel_size, stride, pad,
                             dilation=1, bn=True):
    #depthwise separable convolution
    if bn:
        # 2D batchnorm + relu + depthwise separable convolution
        return nn.Sequential(nn.BatchNorm2D(num_features=in_channels),
                             nn.ReLU(),
                             nn.Conv2D(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=dilation if dilation > 1 else pad,
                                       dilation=dilation,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=False,
                                       groups=in_channels),
                             nn.Conv2D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=False))
    else:
        # 2D relu + depthwise separable convolution
        return nn.Sequential(nn.ReLU(),
                             nn.Conv2D(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=dilation if dilation > 1 else pad,
                                       dilation=dilation,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=False,
                                       groups=in_channels),
                             nn.Conv2D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       weight_attr=nn.initializer.KaimingNormal(),
                                       bias_attr=False))

def refinement1(in_channels, out_channels):
    #color guidance refinement on left image or disparity stage 3
    net = [nn.Conv2D(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     weight_attr=nn.initializer.KaimingNormal(),
                     bias_attr=False
                     )]

    net = net + [preconv2d_depthseperated(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          pad=1,
                                          dilation=2 ** (k + 1)) for k in range(4)]

    return nn.Sequential(*net)

def refinement2(in_channels, out_channels):
    #color guidance refinement on concatenated features
    net = [preconv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=1,
                     pad=1,
                     dilation=8)]

    net = net + [preconv2d_depthseperated(in_channels=out_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          pad=1,
                                          dilation=2 ** k) for k in reversed(range(4))]

    net = net + [nn.Conv2D(in_channels=out_channels,
                           out_channels=1,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           weight_attr=nn.initializer.KaimingNormal(),
                           bias_attr=False
                           )]

    return nn.Sequential(*net)