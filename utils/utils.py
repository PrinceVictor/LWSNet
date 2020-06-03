import paddle.fluid as fluid

def unsqueeze(input, aixs):

    return fluid.layers.unsqueeze(input=input, axes=aixs)

def unsqueeze_repeat_view(input, maxdisp, shape):
    input = fluid.layers.expand(unsqueeze(input, [1]), [1, maxdisp * 2 - 1, 1, 1, 1])
    input = fluid.layers.reshape(input, shape=shape, inplace=True)

    return input