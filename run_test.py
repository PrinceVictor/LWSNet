import paddle
import numpy as np

if __name__ == "__main__":
    conv2d = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=3,
                              out_channels=5,
                              kernel_size=3,
                              stride=1,
                              padding=0),paddle.nn.BatchNorm2D(num_features=5))
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[3, 6, 9], values=[0.1, 0.2, 0.3, 0.4], verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=conv2d.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.uniform([1, 3, 30, 30])
            gt = paddle.uniform([1, 5, 28, 28])
            print(x)
            out = conv2d(x)
            loss = paddle.nn.functional.smooth_l1_loss(out, gt)
            loss.backward()
            sgd.step()
            sgd.clear_gradients()
        scheduler.step()