import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from .maximum_mean_discrepancy import mmd
from tqdm import trange
from panns_inference import AudioTagging
import ipdb


class KernelPannDistanceMaker(object):
	def __init__(self, batch_size=100):
		self.device = 'cuda' if GPU_is_available() else 'cpu'
		self.pann = AudioTagging(checkpoint_path=None, device=self.device)
		self.batch_size = batch_size

	def __call__(self, true_audio, fake_audio):
		kpd = []
		pbar = trange(int(np.ceil(len(true_audio)/self.batch_size)), desc='PANN-Inference')
		for b in pbar:
			true_batch = torch.Tensor(true_audio[b*self.batch_size:(b+1)*self.batch_size])
			_, true_embeddings = self.pann.inference(true_batch)
			fake_batch = torch.Tensor(fake_audio[b*self.batch_size:(b+1)*self.batch_size])
			_, fake_embeddings = self.pann.inference(fake_batch)
			kpd.append(mmd(torch.from_numpy(true_embeddings), torch.from_numpy(fake_embeddings)))
			kpd_mean = np.mean(kpd)
			kpd_std = np.std(kpd)
			pbar.set_description("Kernel Pann Distance = {0:.4f} +- {1:.4f}".format(kpd_mean, kpd_std/2.))
		return {'mean_kpd': np.mean(kpd), 'std_kpd': np.std(kpd)}

