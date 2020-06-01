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
    output = fluid.layers.conv2d(input=input,
                                 num_filters=channel,
                                 filter_size=kernel_size,
                                 stride=stride,
                                 padding=dilation if dilation>1 else padding,
                                 dilation=dilation,
                                 act=None,
                                 param_attr=conv_param_attr,
                                 bias_attr=conv_bias_attr)

    return fluid.layers.batch_norm(input=output,
                                   act=bn_activation,
                                   param_attr=bn_param_attr,
                                   bias_attr=bn_bias_attr,
                                   in_place=True)

def deconvbn(input, channel,
           kernel_size, stride, padding, dilation=1,
           bn_activation=None,
           conv_param_attr=None, conv_bias_attr=None,
           bn_param_attr=None, bn_bias_attr=None):
    output = fluid.layers.conv2d_transpose(input=input,
                                           num_filters=channel,
                                           output_size=None,
                                           filter_size=kernel_size,
                                           padding=padding,
                                           stride=stride,
                                           param_attr=conv_param_attr,
                                           bias_attr=conv_bias_attr,
                                           )
    return fluid.layers.batch_norm(input=output,
                                   act=bn_activation,
                                   param_attr=bn_param_attr,
                                   bias_attr=bn_bias_attr,
                                   in_place=True)

class hourglass():
    def __init__(self, init_channel=8):
        self.init_channel = init_channel

    def conv1(self, input):
        return convbn(input=input,
                      channel=self.init_channel * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      conv_param_attr=layer_init_kaiming_normal(),
                      conv_bias_attr=False,
                      bn_param_attr=layer_init_constant(1.0),
                      bn_bias_attr=layer_init_constant(0.0),
                      bn_activation='relu')

    def conv2(self, input):
        return convbn(input=input,
                      channel=self.init_channel * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      conv_param_attr=layer_init_kaiming_normal(),
                      conv_bias_attr=False,
                      bn_param_attr=layer_init_constant(1.0),
                      bn_bias_attr=layer_init_constant(0.0),
                      bn_activation='relu')

    def conv3(self, input):
        return convbn(input=input,
                      channel=self.init_channel * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      conv_param_attr=layer_init_kaiming_normal(),
                      conv_bias_attr=False,
                      bn_param_attr=layer_init_constant(1.0),
                      bn_bias_attr=layer_init_constant(0.0),
                      bn_activation='relu')

    def conv4(self, input):
        return convbn(input=input,
                      channel=self.init_channel * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      conv_param_attr=layer_init_kaiming_normal(),
                      conv_bias_attr=False,
                      bn_param_attr=layer_init_constant(1.0),
                      bn_bias_attr=layer_init_constant(0.0),
                      bn_activation='relu')

    def conv5(self, input):
        return deconvbn(input=input,
                        channel=self.init_channel*2,
                        kernel_size=3,
                        padding=[1,0,1,0],
                        stride=2,
                        conv_param_attr=layer_init_kaiming_normal(),
                        conv_bias_attr=False,
                        bn_param_attr=layer_init_constant(1.0),
                        bn_bias_attr=layer_init_constant(0.0),
                        bn_activation=None)

    def conv6(self, input):
        return deconvbn(input=input,
                        channel=self.init_channel,
                        kernel_size=3,
                        padding=[1,0,1,0],
                        stride=2,
                        conv_param_attr=layer_init_kaiming_normal(),
                        conv_bias_attr=False,
                        bn_param_attr=layer_init_constant(1.0),
                        bn_bias_attr=layer_init_constant(0.0),
                        bn_activation=None)

    def inference(self, input):
        res = []
        output_0 = self.conv1(input)

        pre = self.conv2(output_0)

        output_1 = self.conv3(pre)

        output_2 = self.conv4(output_1)

        res.append(output_2)

        post = fluid.layers.relu((self.conv5(output_2) + pre))
        res.append(post)

        output = self.conv6(post)
        res.append(output)

        return res

class feature_extraction():

    def __init__(self):
        self.hourglass = hourglass(init_channel=8)

    def dres0(self, input):
        output_0 = convbn(input=input,
                          channel=4,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          dilation=2,
                          conv_param_attr=layer_init_kaiming_normal(),
                          conv_bias_attr=False,
                          bn_param_attr=layer_init_constant(1.0),
                          bn_bias_attr=layer_init_constant(0.0))
        output_0 = fluid.layers.relu(output_0)
        output_1 = convbn(input=output_0,
                          channel=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          dilation=4,
                          conv_param_attr=layer_init_kaiming_normal(),
                          conv_bias_attr=False,
                          bn_param_attr=layer_init_constant(1.0),
                          bn_bias_attr=layer_init_constant(0.0))
        output_1 = fluid.layers.relu(output_1)
        return output_1

    def dres1(self, input):
        output_0 = convbn(input=input,
                          channel=4,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          dilation=2,
                          conv_param_attr=layer_init_kaiming_normal(),
                          conv_bias_attr=False,
                          bn_param_attr=layer_init_constant(1.0),
                          bn_bias_attr=layer_init_constant(0.0))
        output_0 = fluid.layers.relu(output_0)
        output_1 = convbn(input=output_0,
                          channel=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          dilation=2,
                          conv_param_attr=layer_init_kaiming_normal(),
                          conv_bias_attr=False,
                          bn_param_attr=layer_init_constant(1.0),
                          bn_bias_attr=layer_init_constant(0.0))
        return output_1

    def dres2(self, input):
        return self.hourglass.inference(input)

    def dres3(self, input):
        output = convbn(input=input,
                        channel=8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        conv_param_attr=layer_init_kaiming_normal(),
                        conv_bias_attr=False,
                        bn_param_attr=layer_init_constant(1.0),
                        bn_bias_attr=layer_init_constant(0.0),
                        bn_activation='relu')

        return fluid.layers.conv2d(input=output,
                                   num_filters=8,
                                   filter_size=3,
                                   padding=1,
                                   stride=1,
                                   param_attr=layer_init_kaiming_normal(),
                                   bias_attr=False)

    def inference(self, input):

        output_0 = self.dres0(input)

        output_1 = self.dres1(output_0) + output_0

        res = self.dres2(output_1)
        output_2 = res[-1] + output_1

        output_3 = self.dres3(output_2)

        res.pop(-1)
        res.append(output_3)

        return res














