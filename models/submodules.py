import paddle
import paddle.fluid as fluid
import numpy as np
def convbn(input, channel,
           kernel_size, stride, padding, dilation=1,
           conv_activation=None,
           conv_param_attr=None, conv_bias_attr=False,
           bn_param_attr=None, bn_bias_attr=False,):
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
                                   param_attr=bn_param_attr,
                                   bias_attr=bn_bias_attr,
                                   in_place=True)

def hourglass():

def feature_extraction_conv(input):

    def dres0(input):
        output_0 = convbn(input=input,
                          channel=4,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          dilation=2)
        output_0 = fluid.layers.relu(output_0)
        output_1 = convbn(input=output_0,
                          channel=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          dilation=4)
        output_1 = fluid.layers.relu(output_1)
        return output_1

    def dres1(input):
        output_0 = convbn(input=input,
                          channel=4,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          dilation=2)
        output_0 = fluid.layers.relu(output_0)
        output_1 = convbn(input=output_0,
                          channel=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          dilation=2)
        return output_1





