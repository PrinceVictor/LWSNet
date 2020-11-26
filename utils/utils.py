import paddle.fluid as fluid

def unsqueeze(input, aixs):

    return fluid.layers.unsqueeze(input=input, axes=aixs)

def unsqueeze_repeat_view(input, maxdisp, shape):
    input = fluid.layers.expand(unsqueeze(input, [1]), [1, maxdisp * 2 - 1, 1, 1, 1])
    input = fluid.layers.reshape(input, shape=shape)

    return input

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count