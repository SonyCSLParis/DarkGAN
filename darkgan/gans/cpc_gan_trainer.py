import os
import torch
import torch.nn
import torch.nn.functional as F

from .pgan_config import _C
from .cpc_gan import CPCGAN
from .progressive_gan_trainer import ProgressiveGANTrainer
from utils.utils import getMinOccurence, mkdir_in_path, ResizeWrapper
from tqdm import trange

from time import time
import numpy as np

import traceback
import ipdb


class CPCGANTrainer(ProgressiveGANTrainer):
    r"""
    A class managing a progressive GAN training. Logs, chekpoints, visualization,
    and number iterations are managed here.
    """
    _defaultConfig = _C

    def getDefaultConfig(self):
        return ProgressiveGANTrainer._defaultConfig

    def initModel(self):
        r"""
        Initialize the GAN model.
        """
        print("Init P-GAN")
        config = self.initScaleShapes()
        self.model = CPCGAN(useGPU=self.useGPU, **config)

    def init_reference_eval_vectors(self, batch_size=50):
        self.true_ref, self.ref_labels = self.loader.get_validation_set(batch_size)

        if self.modelConfig.ac_gan:
            batch_size = min(batch_size, len(self.ref_labels))
            self.ref_z, _ = self.model.buildNoiseData(batch_size, inputLabels=self.ref_labels, skipAtts=True)
        else:
            self.ref_z, _ = self.model.buildNoiseData(batch_size)

        if self.ref_labels.size(-1) == 33:
            ref_labels1 = self.ref_labels[:, 0]
            ref_labels1 = ref_labels1.unsqueeze(-1).repeat(1, 32).unsqueeze(-1)

            ref_labels2 = self.ref_labels[:, 1:].unsqueeze(-1)

            self.ref_labels = torch.cat([ref_labels1, ref_labels2], dim=-1).reshape(-1, 2)
        elif self.ref_labels.size(-1) == 34:
            ref_labels1 = self.ref_labels[:, 0:2]
            ref_labels1 = ref_labels1.unsqueeze(1).repeat(1, 32, 1)

            ref_labels2 = self.ref_labels[:, 2:].unsqueeze(-1)

            self.ref_labels = torch.cat([ref_labels1, ref_labels2], dim=-1).reshape(-1, 3)
        else:
            self.ref_labels = self.ref_labels.unsqueeze(-1).reshape(-1, 1)

        self.ref_labels_str = self.loader.index_to_labels(self.ref_labels, transpose=True)

        
    def test_GAN(self):
        self.init_reference_eval_vectors()
        # sample fake data
        fake = self.model.test_G(
            input=self.ref_z, getAvG=False, toCPU=not self.useGPU)
        fake_avg = self.model.test_G(
            input=self.ref_z, getAvG=True, toCPU=not self.useGPU)
        
        # predict labels for fake data
        D_fake, fake_emb = self.model.test_D(
            fake, get_labels=self.modelConfig.ac_gan, output_device='cpu')
        if self.modelConfig.ac_gan:
            D_fake = self.loader.index_to_labels(
                D_fake.detach(), transpose=True)

        D_fake_avg, fake_avg_emb = self.model.test_D(
            fake_avg,  get_labels=self.modelConfig.ac_gan, output_device='cpu')
        if self.modelConfig.ac_gan:
            D_fake_avg = self.loader.index_to_labels(
                D_fake_avg.detach(), transpose=True)
        
        # predict labels for true data
        true, _ = self.loader.get_validation_set(
            len(self.ref_z), process=True)
        D_true, true_emb = self.model.test_D(
            true, get_labels=self.modelConfig.ac_gan, output_device='cpu')
        if self.modelConfig.ac_gan:
            D_true = self.loader.index_to_labels(
                D_true.detach(), transpose=True)

        return D_true, true_emb.detach(), \
               D_fake, fake_emb.detach(), \
               D_fake_avg, fake_avg_emb.detach(), \
               true, fake.detach(), fake_avg.detach()


    def run_tests_evaluation_and_visualization(self, scale):
        self.loss_visualizer.publish_cpc_codes(self.ref_labels[:5])
    #     super().run_tests_evaluation_and_visualization(scale)
    # def run_tests_evaluation_and_visualization(self, scale):

        scale_output_dir = mkdir_in_path(self.output_dir, f'scale_{scale}')
        iter_output_dir  = mkdir_in_path(scale_output_dir, f'iter_{self.iter}')
        from utils.utils import saveAudioBatch

        D_true, true_emb, \
        D_fake, fake_emb, \
        D_fake_avg, fake_avg_emb, \
        true, fake, fake_avg = self.test_GAN()

        if self.modelConfig.ac_gan:
            output_dir = mkdir_in_path(iter_output_dir, 'classification_report')
            if not hasattr(self, 'cls_vis'):
                from visualization.visualization import AttClassifVisualizer
                self.cls_vis = AttClassifVisualizer(
                    output_path=output_dir,
                    env=self.modelLabel,
                    save_figs=True,
                    attributes=self.loader.header['attributes'].keys(),
                    att_val_dict=self.loader.header['attributes'])
            self.cls_vis.output_path = output_dir

            D = self.model.getOriginalD()
            if self.ref_labels.size(-1) == 3:
                ref_labels = self.ref_labels.reshape(-1, 32, 3).transpose(1, 2)
                ref_labels = D.downScale(ref_labels, D.inputSizes[D.scale - 1][1]).transpose(1, 2)
                ref_labels = ref_labels.reshape(-1, 3)
            else:
                ref_labels = self.ref_labels.reshape(-1, 32, 2).transpose(1, 2)
                ref_labels = D.downScale(ref_labels, D.inputSizes[D.scale - 1][1]).transpose(1, 2)
                ref_labels = ref_labels.reshape(-1, 2)
            ref_labels = self.loader.index_to_labels(ref_labels, True)

            self.cls_vis.publish(
                ref_labels, 
                D_true,
                name=f'{scale}_true',
                title=f'scale {scale} True data')
            
            self.cls_vis.publish(
                ref_labels, 
                D_fake,
                name=f'{scale}_fake',
                title=f'scale {scale} Fake data')

        if self.save_gen:
            audio_output_dir = mkdir_in_path(iter_output_dir, 'generation')
            saveAudioBatch(
                self.loader.postprocess(fake), 
                path=audio_output_dir, 
                basename=f'gen_audio_scale_{scale}')

        if self.vis_manager != None:
            output_dir = mkdir_in_path(iter_output_dir, 'audio_plots')
            if scale >= self.n_scales -2:
                self.vis_manager.renderAudio = True

            self.vis_manager.set_postprocessing(
                self.loader.get_postprocessor())
            self.vis_manager.publish(
                true[:5], 
                # labels=D_true[:][:5],
                name=f'real_scale_{scale}', 
                output_dir=output_dir)
            self.vis_manager.publish(
                fake[:5], 
                # labels=D_fake[0][:5],
                name=f'gen_scale_{scale}', 
                output_dir=output_dir)


        # inception score
        if 'is' in self.eval_config:
            is_config = self.eval_config['is']
            if not hasattr(self, 'is_maker'):
                from evaluation.metrics.inception_score import InceptionScoreMaker
                is_maker = InceptionScoreMaker(is_config['config_path'])
            iscore = is_maker(audio_output_dir, attribute=is_config['attribute'])
            self.updateRunningLosses(iscore)
        # kernel inception distance
        if 'kid' in self.eval_config:
            kid_config = self.eval_config['kid']
            if not hasattr(self, 'kid_maker'):
                from evaluation.metrics.kid import KernelInceptionDistanceMaker
                self.kid_maker = KernelInceptionDistanceMaker(kid_config['config_path'])
            kid = self.kid_maker(self.loader.postprocess(true), self.loader.postprocess(fake))
            self.updateRunningLosses(kid)
        # kernel PANN distance
        if 'kpd' in self.eval_config:
            if not hasattr(self, 'kpd_maker'):
                from evaluation.metrics.kernel_pann_distance import KernelPannDistanceMaker
                self.kpd_maker = KernelPannDistanceMaker()
            kpd = self.kpd_maker(self.loader.postprocess(true), self.loader.postprocess(fake))
            self.updateRunningLosses(kpd)
        # frechet audio distance
        if 'fad' in self.eval_config:
            try:
                from evaluation.metrics.fad import compute_fad
                true_output_dir = mkdir_in_path('/tmp/', 'generation')
                saveAudioBatch(
                    self.loader.postprocess(true), 
                    path=true_output_dir,
                    overwrite=True,
                    basename=f'audio_scale_{scale}')

                fad_output_dir = mkdir_in_path('/tmp/', 'fad')
                fad = compute_fad(true_output_dir, audio_output_dir, fad_output_dir)
                self.updateRunningLosses({'fad': fad})
            except Exception as e:
                print(f"Cannot compute fad: {e}")
        self.publish_loss()
