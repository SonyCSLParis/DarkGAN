import numpy as np
import click
import ipdb
from utils.utils import *
from panns_inference import AudioTagging, SoundEventDetection, labels
from pathlib import Path
from tqdm import trange, tqdm
from data.audio_transforms import loader
from multiprocessing import Pool, cpu_count
import resource
from functools import partial
import pickle as pkl
from datetime import datetime


# Cnn14_mAP=0.431.pth sample-rate
SAMPLE_RATE = 32000

def parallel_load(paths):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
    p = Pool(cpu_count())
    out = list(p.map(partial(loader, 32000, 32000), paths))
    p.close()
    return out

def parse_conditional_atts(paths):
	cond_atts = []
	for path in paths:
		with open(path, 'rb') as file:
			# hack! [:-26] remove pitch attributes
			cond_atts.append(pkl.load(file)[:-26])
		file.close()
	return cond_atts


def discrete_range_coherence(attribute, result, h_range, att_pred):
	result[attribute] = {'E1': 0, 'E2': 0, 'E3': 0}
	h_range = int(np.floor(len(att_pred)/n_tests))
	for i in trange(h_range, desc='Checking hypothesis'):
		# check E1: low < high
		if att_pred[i*n_tests] < att_pred[i*n_tests + 6]:
			result[attribute]['E1'] += 1
		# check E2: mid < high
		if  att_pred[i*n_tests + 4] < att_pred[i*n_tests + 6]:
			result[attribute]['E2'] += 1
		# check E3: low < mid
		if  att_pred[i*n_tests] < att_pred[i*n_tests + 4]:
			result[attribute]['E3'] += 1
	result[attribute]['E1'] /= h_range
	result[attribute]['E2'] /= h_range
	result[attribute]['E3'] /= h_range
	return result


def correlation_coherence(attribute, result, params_in, att_pred):
	result[attribute] = {'avg_corrcoef': 0, 'avg_cov': 0, 'avg_cov_real': 0}
	params_in = torch.Tensor(params_in).transpose(1, 0)
	att_pred = torch.Tensor(att_pred).transpose(1, 0)
	for i in range(len(att_pred)):
		# hack to check random corr (keep commented)
		# params_in[i] = np.random.uniform(0, 1, size=len(att_pred[i]))
		result[attribute]['avg_corrcoef'] += np.corrcoef(params_in[i], att_pred[i])[0][1]
		result[attribute]['avg_cov'] += np.cov(params_in[i], att_pred[i])[0][1]
		result[attribute]['avg_cov_real'] += np.cov(params_in[i], params_in[i])[0][1]
		break
	# result[attribute]['avg_corrcoef'] /= len(att_pred)
	# result[attribute]['avg_cov'] /= len(att_pred)
	# result[attribute]['avg_cov_real'] /= len(att_pred)
	return result


@click.command()
@click.option('-d', '--dir', type=click.Path(exists=True), required=True, default='')
@click.option('-t', '--test', type=str, required=True, default='range')
def main(dir, test):
	root_path = Path(dir)
	device = 'cuda' if GPU_is_available() else 'cpu'
	# hardcoded for now
	n_tests = 10
	batch_size = 100

	# audioset classifier
	at = AudioTagging(checkpoint_path=None, device=device)

	# output dict
	result = {}

	# prepare paths
	folders = [f for f in root_path.iterdir() if f.is_dir()]

	# get attributes used for generatiuon
	gan_att_list = sorted([f.name for f in folders])

	# get indexes of the attributes used for gen from al those predicted by the audioset classifier
	att_index = [labels.index(a) for a in gan_att_list]
	pbar = tqdm(sorted(folders))

	# filenames, params_in = parse_conditional_atts()
	folder_n = 0
	for att_folder in pbar:
		attribute = att_folder.name
		pbar.set_description(attribute)
		audio_paths = sorted(att_folder.glob('*.wav'))
		cond_atts_paths = sorted(att_folder.glob('*.pkl'))

		# check ordering
		assert all([a.stem == b.stem for a, b in zip(audio_paths, cond_atts_paths)]), \
			'Error! Ordering of attribute pkl file paths and audio paths does not match'

		# read pkl files with the conditional atts used for generation
		cond_atts = parse_conditional_atts(cond_atts_paths)

		# read audio
		audios = torch.Tensor(parallel_load(audio_paths))

		# inference
		pred = []
		for b in trange(int(np.ceil(len(audios)/batch_size)), desc='Inference'):
			pred.append(at.inference(audios[b*batch_size:(b+1)*batch_size])[0])
		pred = np.concatenate(pred, axis=0)

		if test == 'range':
			# take predictions of the attribute under consideration
			att_pred = pred[:, labels.index(attribute)]
			# run test
			result = discrete_range_coherence(attribute, result, h_range, att_pred)

		elif test == 'corr':
			result = correlation_coherence(attribute, result, cond_atts, pred[:, att_index])

		print(f'Attribute {attribute}: {result[attribute]}')
	save_json(result, root_path.joinpath(f'evaluation_{test}_{datetime.now().strftime("%H:%M:%S")}.json'))
if __name__ == '__main__':
	main()

