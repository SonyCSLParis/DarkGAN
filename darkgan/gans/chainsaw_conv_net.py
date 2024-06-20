import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import 

from .progressive_conv_net import GNet as ProgressiveGNet

from utils.utils import num_flat_features
from.mini_batch_stddev_module import miniBatchStdDev
import ipdb


class GNet(ProgressiveGNet):

    def __init__(self, **kargs):
        syper(ProgressiveGNet, self).__init__(**kargs)

    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 x scalesDepth[0]
        layer.
        """
        ipdb.set_trace()
        self.dimLatent = dimLatentVector

        self.formatLayer = ChainSawGANInitFormatLayer(self.dimLatent,
                                           self.sizeScale0[0] * self.sizeScale0[1] * self.scalesDepth[0], #Change factor for other than 8
                                           equalized=self.equalizedlR,
                                           initBiasToZero=self.initBiasToZero)

