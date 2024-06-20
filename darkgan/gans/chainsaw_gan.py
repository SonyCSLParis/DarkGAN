import torch.optim as optim
import torch.nn as nn

from .progressive_gan import ProgressiveGAN

from .progressive_autoencoder_conv_net import AEGNet, DNet

from torch.nn import DataParallel
from .gradient_losses import WGANGPGradientPenalty
from utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible, GPU_is_available
from torch import randperm
import torch

import numpy as np
import ipdb


class ChainSawGAN(ProgressiveGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """
    def buildNoiseData(self, n_samples, inputLabels=None, skipAtts=False):
        r"""
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """
        # sample random vector of length n_samples
        inputLatent = torch.randn(n_samples, self.config.noiseVectorDim).to(self.device)
        # HACK:
        # if skipAtts and all(k in self.ClassificationCriterion.skipAttDfake \
        #         for k in self.ClassificationCriterion.keyOrder):
        #     return inputLatent, None
        #################
        
        if self.config.attribKeysOrder and self.config.ac_gan:
            if inputLabels is not None:
                latentRandCat = self.ClassificationCriterion.buildLatentCriterion(inputLabels, skipAtts=skipAtts)
                targetRandCat = inputLabels
            else:
                targetRandCat, latentRandCat = \
                    self.ClassificationCriterion.buildRandomCriterionTensor(n_samples, skipAtts)

            targetRandCat = targetRandCat.to(self.device)
            latentRandCat = latentRandCat.to(self.device)
            inputLatent = torch.cat((inputLatent, latentRandCat), dim=1)

            return inputLatent, targetRandCat
        return inputLatent, None