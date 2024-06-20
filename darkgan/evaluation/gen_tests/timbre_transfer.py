import os

from datetime import datetime

from utils.utils import *
from data.preprocessing import AudioProcessor
from gans.ac_criterion import ACGANCriterion
import numpy as np
import torch
import random
from data.loaders import get_data_loader
from data.audio_transforms import loader
import ipdb
from tqdm import tqdm, trange
import pickle as pkl
import click
import ipdb
from panns_inference import AudioTagging, SoundEventDetection, labels
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_audios_in_dir_and_predict(path, T, audio_length):
    audioset_clf = AudioTagging()
    files = list_files_abs_path(path)
    audio = [loader(x=f, sample_rate=16000, audio_length=audio_length) for f in files]
    audio = [list(f) + [0]*(audio_length - len(f)) for f in audio]
    # audio = list(map(lambda x: x + [0]*(audio_length - len(x)), audio))
    ipdb.set_trace()
    pred = audioset_clf.inference(torch.Tensor(audio), T=T)[0]

    return audio, pred


@click.command()
@click.option('-d', '--model_dir', type=click.Path(exists=True), required=True, default='')
@click.option('--iteration', type=int, required=False, default=None)
@click.option('--scale', type=int, required=False, default=None)
@click.option('-o', '--output_dir', type=click.Path(), required=False, default='')
@click.option('-t', '--true_dir', type=click.Path(), required=False, default='')
def main(model_dir, iteration, scale, output_dir, true_dir):
    model, config, model_name = load_model_checkp(dir=model_dir, iter=iteration, scale=scale)

    latentDim = model.config.categoryVectorDim_G
    device = 'cuda' if GPU_is_available() else 'cpu'

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']
    # loader_config['criteria']['audioset']['temperature'] = 1

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)

    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = audioset_dict['values']
    as_att_idx = [labels.index(att) for att in audioset_keys]
    true_audios, pred = read_audios_in_dir_and_predict(true_dir, T=loader_config['criteria']['audioset']['temperature'], audio_length=processor.audio_length)
    pred = pred[:, as_att_idx]

    # n_gen = 50000
    _, val_feats = dummy_loader.get_validation_set(50000)

    n_gen = len(pred)
    train_feats = torch.Tensor(dummy_loader.metadata)
    
    print(f"Number of instances = {n_gen}")

    all_feats = torch.cat([train_feats, val_feats], dim=0)
    all_audios = dummy_loader.data + dummy_loader.val_data


    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"timbre_transfer")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))


    # rand_index = torch.randint(len(all_audios), (n_gen,))
    # labels_in = all_feats[rand_index]
    # true_audios = np.array([all_audios[i] for i in rand_index])
    true_audios = map(postprocess, true_audios)

    labels_in[:, :pred.shape[1]] = torch.Tensor(pred)
    for j, l in enumerate(labels_in):
        # sample from G using rand labels
        z_rand, _ = model.buildNoiseData(n_gen, inputLabels=l.repeat(n_gen, 1), skipAtts=True)
        # sample from G
        rand_gen_batch = model.test(z_rand.to(device), toCPU=not GPU_is_available()).cpu()
        # postprocess output
        audio_out = map(postprocess, rand_gen_batch)
        audio_out = torch.Tensor(list(map(lambda x: (x - x.mean())/(x.max() - x.mean()), audio_out)))
        audio_out = torch.cat([audio_out, torch.zeros((n_gen, 2048))], dim=1)

        saveAudioBatch(audio_out.flatten().reshape((1, -1)),
               path=output_dir,
               basename=f'{j}_sample',
               sr=config["transform_config"]["sample_rate"])
    saveAudioBatch(true_audios,
               path=output_dir,
               basename=f'true_sample',
               sr=config["transform_config"]["sample_rate"])

if __name__ == "__main__":
    main()