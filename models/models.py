import paddle
import paddle.fluid as fluid

from models.submodules import *

class Ownnet():
    def __init__(self):
        self.feature_extraction = feature_extraction()

    def feature_extractor(self, input):
        return self.feature_extraction.inference(input)

    def inference(self, input):

        output = self.feature_extractor(input)

        return output