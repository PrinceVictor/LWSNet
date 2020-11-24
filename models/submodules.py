import paddle
import paddle.fluid as fluid
import numpy as np

def layer_init_constant(value=0.0):
    return fluid.initializer.ConstantInitializer(value=value)

def layer_init_kaiming_normal():
    return fluid.initializer.MSRAInitializer(uniform=False)

def convbn(input, channel,
           kernel_size, stride, padding, dilation=1,
           bn_activation=None,
           conv_param_attr=None, conv_bias_attr=None,
           bn_param_attr=None, bn_bias_attr=None):

    return fluid.dygraph.Sequential(fluid.dygraph.Conv2D(num_channels=input,
                                                         num_filters=channel,
                                                         filter_size=kernel_size,
                                                         stride=stride,
                                                         padding=dilation if dilation>1 else padding,
                                                         dilation=dilation,
                                                         act=None,
                                                         param_attr=conv_param_attr,
                                                         bias_attr=conv_bias_attr),
                                    fluid.dygraph.BatchNorm(num_channels=channel,
                                                            act=bn_activation,
                                                            param_attr=bn_param_attr,
                                                            bias_attr=bn_bias_attr,
                                                            in_place=False))




def deconvbn(input, channel,
             kernel_size, stride, padding, dilation=1,
             bn_activation=None,
             conv_param_attr=None, conv_bias_attr=None,
             bn_param_attr=None, bn_bias_attr=None):
    return fluid.dygraph.Sequential(fluid.dygraph.Conv2DTranspose(num_channels=input,
                                                                  num_filters=channel,
                                                                  output_size=(None),
                                                                  filter_size=kernel_size,
                                                                  padding=padding,
                                                                  stride=stride,
                                                                  param_attr=conv_param_attr,
                                                                  bias_attr=conv_bias_attr),
                                    fluid.dygraph.BatchNorm(num_channels=channel,
                                                            act=bn_activation,
                                                            param_attr=bn_param_attr,
                                                            bias_attr=bn_bias_attr,
                                                            in_place=False))

def batch_relu_conv3d(input, channel,
                      kernel_size=3, stride=1, padding=1, bn3d=True,
                      bn_activation='relu',
                      conv_param_attr=None, conv_bias_attr=None,
                      bn_param_attr=None, bn_bias_attr=None):

    if bn3d:
        return fluid.dygraph.Sequential(fluid.dygraph.BatchNorm(num_channels=input,
                                                                act='relu',
                                                                param_attr=bn_param_attr,
                                                                bias_attr=bn_bias_attr,
                                                                in_place=False),
                                        fluid.dygraph.Conv3D(num_channels=input,
                                                             num_filters=channel,
                                                             filter_size=kernel_size,
                                                             padding=padding,
                                                             stride=stride,
                                                             param_attr=conv_param_attr,
                                                             bias_attr=conv_bias_attr))
    else:
        return fluid.dygraph.Sequential(fluid.layers.relu(),
                                        fluid.dygraph.Conv3D(num_channels=input,
                                                             num_filters=channel,
                                                             filter_size=kernel_size,
                                                             padding=padding,
                                                             stride=stride,
                                                             param_attr=conv_param_attr,
                                                             bias_attr=conv_bias_attr))

def preconv2d(input, channel,kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return fluid.dygraph.Sequential(fluid.dygraph.BatchNorm(num_channels=input,
                                                                act='relu',
                                                                in_place=False),
                                        fluid.dygraph.Conv2D(num_channels=input,
                                                             num_filters=channel,
                                                             filter_size=kernel_size,
                                                             stride=stride,
                                                             padding=dilation if dilation > 1 else pad,
                                                             dilation=dilation,
                                                             param_attr=layer_init_kaiming_normal(),
                                                             bias_attr=False))



def preconv2d_depthseperated(input, channel,
                             kernel_size, stride, pad,
                             dilation=1, bn=True, seperated=False):
    if bn:
        return fluid.dygraph.Sequential(fluid.dygraph.BatchNorm(num_channels=input,
                                                                act='relu',
                                                                in_place=False),
                                        fluid.dygraph.Conv2D(num_channels=input,
                                                             num_filters=channel,
                                                             filter_size=kernel_size,
                                                             stride=stride,
                                                             padding=dilation if dilation > 1 else pad,
                                                             dilation=dilation,
                                                             param_attr=layer_init_kaiming_normal(),
                                                             bias_attr=False,
                                                             groups=input),
                                        fluid.dygraph.Conv2D(num_channels=channel,
                                                             num_filters=channel,
                                                             filter_size=1,
                                                             stride=1,
                                                             padding=0,
                                                             dilation=1,
                                                             param_attr=layer_init_kaiming_normal(),
                                                             bias_attr=False))
    else:
        return fluid.dygraph.Sequential(fluid.layers.relu(),
                                        fluid.dygraph.Conv2D(num_channels=input,
                                                             num_filters=channel,
                                                             filter_size=kernel_size,
                                                             stride=stride,
                                                             padding=dilation if dilation > 1 else pad,
                                                             dilation=dilation,
                                                             param_attr=layer_init_kaiming_normal(),
                                                             bias_attr=False,
                                                             groups=input),
                                        fluid.dygraph.Conv2D(num_channels=channel,
                                                             num_filters=channel,
                                                             filter_size=1,
                                                             stride=1,
                                                             padding=0,
                                                             dilation=1,
                                                             param_attr=layer_init_kaiming_normal(),
                                                             bias_attr=False))

# class Post_3DConvs(fluid.dygraph.Layer):
#     def __init__(self, layers, channels):
#         super(Post_3DConvs, self).__init__()
#         self.layers = layers
#         self.channels = channels
#
#         self.layer1 = batch_relu_conv3d(input=1,
#                                         channel=channels,
#                                         conv_param_attr=layer_init_kaiming_normal(),
#                                         conv_bias_attr=False,
#                                         bn_param_attr=layer_init_constant(1.0),
#                                         bn_bias_attr=layer_init_constant(0.0))
#         self.layer_list = []
#         for i in range(self.layers):
#             temp_layer = batch_relu_conv3d(input=channels,
#                                            channel=channels,
#                                            conv_param_attr=layer_init_kaiming_normal(),
#                                            conv_bias_attr=False,
#                                            bn_param_attr=layer_init_constant(1.0),
#                                            bn_bias_attr=layer_init_constant(0.0))
#             self.layer_list.append(temp_layer)
#         self.layer_list = fluid.dygraph.LayerList(self.layer_list)
#
#         self.layer2 = batch_relu_conv3d(input=channels,
#                                         channel=1,
#                                         conv_param_attr=layer_init_kaiming_normal(),
#                                         conv_bias_attr=False,
#                                         bn_param_attr=layer_init_constant(1.0),
#                                         bn_bias_attr=layer_init_constant(0.0))
#
#     def forward(self, input):
#         output = self.layer1(input)
#
#         for i in range(self.layers):
#             output = self.layer_list[i](output)
#
#         output = self.layer2(output)
#
#         return output


def Post_3DConvs(layers, channels):
        output = [batch_relu_conv3d(input=1,
                                    channel=channels,
                                    conv_param_attr=layer_init_kaiming_normal(),
                                    conv_bias_attr=False,
                                    bn_param_attr=layer_init_constant(1.0),
                                    bn_bias_attr=layer_init_constant(0.0))]
        for i in range(layers):
            output = output + [batch_relu_conv3d(input=channels,
                                                 channel=channels,
                                                 conv_param_attr=layer_init_kaiming_normal(),
                                                 conv_bias_attr=False,
                                                 bn_param_attr=layer_init_constant(1.0),
                                                 bn_bias_attr=layer_init_constant(0.0))]
        output = output + [batch_relu_conv3d(input=channels, channel=1, conv_param_attr=layer_init_kaiming_normal())]
        return fluid.dygraph.Sequential(*output)

def refinement1(in_channels, out_channels):

    output = [fluid.dygraph.Conv2D(num_channels=in_channels,
                                   num_filters=out_channels,
                                   filter_size=3,
                                   stride=1,
                                   padding=1,
                                   param_attr=layer_init_kaiming_normal(),
                                   bias_attr=layer_init_kaiming_normal())]

    for k in range(4):
        output = output + [preconv2d_depthseperated(input=out_channels,
                                                    channel=out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    pad=1,
                                                    dilation=2 ** (k + 1))]

    return fluid.dygraph.Sequential(*output)

def refinement2(in_channels, out_channels):

    output = [preconv2d(input=in_channels,
                        channel=out_channels,
                        kernel_size=3,
                        stride=1,
                        pad=1,
                        dilation=8)]

    for k in reversed(range(3)):
        output = output + [preconv2d_depthseperated(input=out_channels,
                                                    channel=out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    pad=1,
                                                    dilation=2 ** k)]

    output = output + [fluid.dygraph.Conv2D(num_channels=out_channels,
                                            num_filters=1,
                                            filter_size=3,
                                            stride=1,
                                            padding=1,
                                            param_attr=layer_init_kaiming_normal(),
                                            bias_attr=layer_init_kaiming_normal())]

    return fluid.dygraph.Sequential(*output)

class hourglass(fluid.dygraph.Layer):
    def __init__(self, init_channel=8):
        super(hourglass, self).__init__()
        self.init_channel = init_channel

        self.conv1 = fluid.dygraph.Sequential(convbn(input=self.init_channel,
                                                     channel=self.init_channel * 2,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation='relu'))

        self.conv2 = fluid.dygraph.Sequential(convbn(input=self.init_channel * 2,
                                                     channel=self.init_channel * 2,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation='relu'))

        self.conv3 = fluid.dygraph.Sequential(convbn(input=self.init_channel * 2,
                                                     channel=self.init_channel * 2,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation='relu'))

        self.conv4 = fluid.dygraph.Sequential(convbn(input=self.init_channel * 2,
                                                     channel=self.init_channel * 2,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation='relu'))

        self.conv5 = fluid.dygraph.Sequential(deconvbn(input=self.init_channel * 2,
                                                       channel=self.init_channel * 2,
                                                       kernel_size=3,
                                                       padding=(1, 1),
                                                       stride=2,
                                                       conv_param_attr=layer_init_kaiming_normal(),
                                                       conv_bias_attr=False,
                                                       bn_param_attr=layer_init_constant(1.0),
                                                       bn_bias_attr=layer_init_constant(0.0),
                                                       bn_activation=None))

        self.conv6 = fluid.dygraph.Sequential(deconvbn(input=self.init_channel * 2,
                                                       channel=self.init_channel,
                                                       kernel_size=3,
                                                       padding=(1, 1),
                                                       stride=2,
                                                       conv_param_attr=layer_init_kaiming_normal(),
                                                       conv_bias_attr=False,
                                                       bn_param_attr=layer_init_constant(1.0),
                                                       bn_bias_attr=layer_init_constant(0.0),
                                                       bn_activation=None))

    def forward(self, input):
        res = []
        output = self.conv1(input)

        pre = self.conv2(output)

        output = self.conv3(pre)

        output = self.conv4(output)
        res.append(output)

        post = fluid.layers.relu((fluid.layers.pad2d(self.conv5(output), paddings=[1,0,1,0]) + pre))
        res.append(post)

        output = self.conv6(post)
        res.append(fluid.layers.pad2d(output, paddings=[1,0,1,0]))

        return res

class feature_extraction(fluid.dygraph.Layer):

    def __init__(self):
        super(feature_extraction, self).__init__()

        self.dres0 = fluid.dygraph.Sequential(convbn(input=3,
                                                     channel=4,
                                                     kernel_size=3,
                                                     stride=2,
                                                     padding=1,
                                                     dilation=2,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation="relu"),
                                              convbn(input=4,
                                                     channel=8,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=4,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation="relu"))

        self.dres1 = fluid.dygraph.Sequential(convbn(input=8,
                                                     channel=4,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=2,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation="relu"),
                                              convbn(input=4,
                                                     channel=8,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=2,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation=None))

        self.dres2 = hourglass(init_channel=8)

        self.dres3 = fluid.dygraph.Sequential(convbn(input=8,
                                                     channel=8,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     conv_param_attr=layer_init_kaiming_normal(),
                                                     conv_bias_attr=False,
                                                     bn_param_attr=layer_init_constant(1.0),
                                                     bn_bias_attr=layer_init_constant(0.0),
                                                     bn_activation="relu"),
                                              fluid.dygraph.Conv2D(num_channels=8,
                                                                   num_filters=8,
                                                                   filter_size=3,
                                                                   padding=1,
                                                                   stride=1,
                                                                   param_attr=layer_init_kaiming_normal(),
                                                                   bias_attr=False))


    def forward(self, input):

        output = self.dres0(input)

        output = self.dres1(output) + output

        res = self.dres2(output)

        output = res[-1] + output

        output = self.dres3(output)

        res.pop(-1)
        res.append(output)

        return res














