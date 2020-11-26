import numpy as np

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

if __name__ == "__main__":
    print(__imagenet_stats["mean"])
    print(__imagenet_stats)