import torch.optim as optim
import torch.nn as nn

from .progressive_gan import ProgressiveGAN
from utils.config import BaseConfig
from.progressive_conv_net import DNet as FCDNet
from .cpc_conv_net import DNet, GNet
from .gdpp_loss import GDPPLoss

from torch.nn import DataParallel
from .gradient_losses import WGANGPGradientPenalty
from utils.utils import loadPartOfStateDict, finiteCheck, \
    loadStateDictCompatible, GPU_is_available
from torch import randperm
import torch

import numpy as np
import ipdb


class CPCGAN(ProgressiveGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """
    def getNetG(self):
        print("CPC-GAN: Building Generator")
        gnet = GNet(dimLatent=self.config.latentVectorDim,
                          depthScale0=self.config.depthScale0,
                          initBiasToZero=self.config.initBiasToZero,
                          leakyReluLeak=self.config.leakyReluLeak,
                          normalization=self.config.perChannelNormalization,
                          generationActivation=self.lossCriterion.generationActivation,
                          dimOutput=self.config.dimOutput,
                          equalizedlR=self.config.equalizedlR,
                          sizeScale0=self.config.sizeScale0,
                          transposed=self.config.transposed,
                          scaleSizes=self.config.scaleSizes,
                          formatLayerType=self.config.formatLayerType)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):
        dnet = DNet(self.config.depthScale0D,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    # self.config.categoryVectorDim,
                    self.ClassificationCriterion.attribSize[2],
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR,
                    sizeScale0=self.config.sizeScale0,
                    inputSizes=self.config.scaleSizes)

        dnet2 = FCDNet(self.config.depthScale0D,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    self.ClassificationCriterion.attribSize[0] +
                    self.ClassificationCriterion.attribSize[1],
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR,
                    sizeScale0=self.config.sizeScale0,
                    inputSizes=self.config.scaleSizes)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)
            dnet2.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)
            dnet2.setNewAlpha(self.config.alpha)

        self.netD_global = dnet2
        return dnet

    def updateSolversDevice(self, buildAvG=True):
        super().updateSolversDevice(buildAvG)
        if not isinstance(self.netD_global, nn.DataParallel) and self.useGPU:
            self.netD_global = nn.DataParallel(self.netD_global)
        self.netD_global.to(self.device)
        self.optimizerD2 = self.getOptimizerD()
        self.optimizerD2.zero_grad()

    def addScale(self, depthNewScale):
        self.netD_global = self.getOriginalD2()
        # Hack to allow different depthScales in G and D
        if type(depthNewScale) is list:
            self.netD_global.addScale(depthNewScale[1])
        else:
            self.netD_global.addScale(depthNewScale)
        super().addScale(depthNewScale)

    def test_D(self, input, get_labels=True, get_embeddings=True, output_device=torch.device('cpu')):
        input = input.to(self.device)
        pred, embedding = self.netD(input, True)
        pred2, embedding2 = self.netD_global(input, True)

        pred2 = pred2.unsqueeze(1).repeat(1, pred.size(-1), 1)
        pred = pred.transpose(1, 3)
        pred = pred.reshape(-1, pred.size(-1))
        pred2 = pred2.reshape(-1, pred2.size(-1))
        pred = torch.cat([pred2[:, :-1], pred], dim=1)

        if get_labels:
            pred, _ = self.ClassificationCriterion.getPredictionLabels(pred)
        if get_embeddings:
            return pred.detach().to(output_device), embedding.detach().to(output_device)
        else:
            return pred.detach().to(output_device)

    def getOriginalD2(self):
        r"""
        Retrieve the original G network. Use this function
        when you want to modify G after the initialization
        """
        if isinstance(self.netD_global, nn.DataParallel):
            return self.netD_global.module
        return self.netD_global

    def getOptimizerD(self):
        # ipdb.set_trace()
        self.config.lrD = self.config.learning_rate
        if type(self.config.learning_rate) is list:
            self.config.lrD = self.config.learning_rate[1]
        return optim.Adam(filter(lambda p: p.requires_grad, list(self.netD.parameters()) + list(self.netD_global.parameters())),
                          betas=[0, 0.99], lr=self.config.lrD)


    def buildNoiseData(self, n_samples, inputLabels=None, skipAtts=False, test=False):
        r"""
        Build a batch of latent vectors for the generator.

        Args:
            n_samples (int): number of vector in the batch
        """
        # sample random vector of length n_samples
        inputLatent = torch.randn(n_samples, int(self.config.noiseVectorDim)).to(self.device)
        inputLatent = inputLatent[:, :, None, None]

        if self.config.attribKeysOrder and self.config.ac_gan:
            assert inputLabels is not None, "CPC-GAN needs code sequence conditioning"

            # IT ASSUMES INST & PITCH CONDITIONING
            # assert inputLabels.size(-1) == 34, "Error in experiment config!!@#$#@$"
            pitch_inst = inputLabels[:, :2]
            cpc_seq = inputLabels[:, 2:]
            D = self.getOriginalD()
            # Downsample cpc_seq
            if not test:
                cpc_seq = D.downScale(cpc_seq.unsqueeze(1).unsqueeze(1), (1, D.inputSizes[D.scale - 1][1]))
            # repeat global conditioning for all seq steps
            global_cond = pitch_inst.unsqueeze(1).repeat(1, cpc_seq.size(-1), 1)
            input_labels = torch.cat([global_cond.reshape(-1, 2), cpc_seq.reshape((-1, 1))], dim=-1)

            latentRandCat = self.ClassificationCriterion.buildLatentCriterion(
                input_labels, 
                skipAtts=skipAtts)
            latentRandCat = latentRandCat.reshape(n_samples, -1, latentRandCat.size(-1))
            latentRandCat = latentRandCat.transpose(1, 2).unsqueeze(2)

            targetRandCat = inputLabels
            inputLatent = inputLatent.repeat((1, 1, 1, cpc_seq.size(-1)))

            targetRandCat = targetRandCat.to(self.device)
            latentRandCat = latentRandCat.to(self.device)

            inputLatent = torch.cat((inputLatent, latentRandCat), dim=1)

            return inputLatent, input_labels
        return inputLatent, None

    def optimizeD(self, allLosses):
        batch_size = self.real_input.size(0)
        self.optimizerD.zero_grad()

        latent_vector, true_labels = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)
        # generate fake batch and detach (we're not updating G)
        fake_batch = self.netG(latent_vector).detach()

        # #1 Real data
        pred_true = self.netD(self.real_input)
        pred_true2 = self.netD_global(self.real_input)
        
        pred_true2 = pred_true2.unsqueeze(1).repeat(1, pred_true.size(-1), 1).reshape(-1, pred_true2.size(-1))
        
        pred_fake = self.netD(fake_batch, False)
        pred_fake = pred_fake.transpose(1, 3)
        pred_fake = pred_fake.reshape(-1, pred_fake.size(-1))

        pred_fake2 = self.netD_global(fake_batch, False)
        pred_fake2 = pred_fake2.unsqueeze(1).repeat(1, pred_true.size(-1), 1).reshape(-1, pred_fake2.size(-1))


        # CLASSIFICATION LOSS
        if self.config.ac_gan:
            # Classification criterion for True and Fake data
            # predRealD = predRealD.sum(-1)
            pred_true = pred_true.transpose(1, 3)
            pred_true = pred_true.reshape(-1, pred_true.size(3))
            pred_true = torch.cat([pred_true2[:, :-1], pred_true], dim=1)

            if self.config.weightConditionD > 0:
                allLosses["lossD_classif"] = \
                    self.classificationPenalty(pred_true,
                                               true_labels,
                                               self.config.weightConditionD,
                                               backward=True)

        # OBJECTIVE FUNCTION FOR TRUE AND FAKE DATA
        lossD = self.lossCriterion.getCriterion(pred_true, True)
        lossD2 = self.lossCriterion.getCriterion(pred_true2, True)

        allLosses["lossD_real"] = lossD.item()
        allLosses["lossD_real2"] = lossD2.item()

        lossDFake = self.lossCriterion.getCriterion(pred_fake, False)
        lossDFake2 = self.lossCriterion.getCriterion(pred_fake2, False)

        allLosses["lossD_fake"] = lossDFake.item()
        allLosses["lossD_fake2"] = lossDFake2.item()
        
        lossD += (lossDFake + lossDFake2)/2

        # #3 WGAN Gradient Penalty loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_GP"], allLosses["lipschitz_norm"] = \
                WGANGPGradientPenalty(input=self.real_input,
                                        fake=fake_batch,
                                        discriminator=self.netD,
                                        weight=self.config.lambdaGP,
                                        backward=True)

            allLosses["lossD_GP2"], allLosses["lipschitz_norm2"] = \
                WGANGPGradientPenalty(input=self.real_input,
                                        fake=fake_batch,
                                        discriminator=self.netD_global,
                                        weight=self.config.lambdaGP,
                                        backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = ((pred_true[:, -1] ** 2).sum()+(pred_true2[:, -1] ** 2).sum())*0.5 * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()

        # # 5 Logistic gradient loss
        if self.config.logisticGradReal > 0:
            allLosses["lossD_logistic"] = \
                logisticGradientPenalty(self.real_input, self.netD,
                                        self.config.logisticGradReal,
                                        backward=True)
        lossD.backward()

        # self.register_D_grads()
        # finiteCheck(self.netD.module.parameters())
        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        # Logs
        lossD = 0
        for key, val in allLosses.items():

            if key.find("lossD") == 0:
                lossD += val

        allLosses["lossD"] = lossD

        return allLosses  

    def optimizeG(self, allLosses):
        batch_size = self.real_input.size(0)
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # #1 Image generation
        inputLatent, true_labels = self.buildNoiseData(batch_size, self.realLabels, skipAtts=True)
        fake_batch = self.netG(inputLatent)

        # #2 Status evaluation
        pred_fake, fake_emb  = self.netD(fake_batch, True)
        pred_fake2, fake_emb2 = self.netD_global(fake_batch, True)
        pred_fake2 = pred_fake2.unsqueeze(1).repeat(1, pred_fake.size(-1), 1).reshape(-1, pred_fake2.size(-1))
        
        # #2 Classification criterion
        if self.config.ac_gan:
            pred_fake = pred_fake.transpose(1, 3)
            pred_fake = pred_fake.reshape(-1, pred_fake.size(3))
            pred_fake = torch.cat([pred_fake2[:, :-1], pred_fake], dim=1)

            if self.config.weightConditionG > 0:
                G_classif_fake = \
                    self.classificationPenalty(pred_fake,
                                               true_labels,
                                               self.config.weightConditionG,
                                               backward=True,
                                               skipAtts=True)
                allLosses["lossG_classif"] = G_classif_fake
        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(pred_fake, True)
        allLosses["lossG_fake"] = lossGFake.item()
        allLosses["Spread_R-F"] = allLosses["lossD_real"] - allLosses["lossG_fake"]

        if self.config.GDPP:
            _, real_emb = self.netD(self.real_input, True)
            _, real_emb2 = self.netD_global(self.real_input, True)

            fake_emb = fake_emb.transpose(1, 3)
            real_emb = real_emb.transpose(1, 3)
            fake_emb = fake_emb.reshape(-1, fake_emb.size(-1))
            real_emb = real_emb.reshape(-1, real_emb.size(-1))

            allLosses["GDPP"] = GDPPLoss(fake_emb, real_emb)
            allLosses["GDPP2"] = GDPPLoss(fake_emb2, real_emb2)

        # Back-propagate generator losss
        lossGFake.backward()
        finiteCheck(self.getOriginalG().parameters())
        self.register_G_grads()
        self.optimizerG.step()

        lossG = 0
        for key, val in allLosses.items():

            if key.find("lossG") == 0:
                lossG += val

        allLosses["lossG"] = lossG

        # Update the moving average if relevant
        if isinstance(self.avgG, nn.DataParallel):
            avgGparams = self.avgG.module.parameters()
        else:
            avgGparams = self.avgG.parameters()       
        
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            avgGparams):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return allLosses
