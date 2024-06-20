import os

from datetime import datetime

from utils.utils import *
from data.preprocessing import AudioProcessor
from gans.ac_criterion import ACGANCriterion
import numpy as np
import torch
import random
from data.loaders import get_data_loader
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

@click.command()
@click.option('-d', '--model_dir', type=click.Path(exists=True), required=True, default='')
@click.option('--iteration', type=int, required=False, default=None)
@click.option('--scale', type=int, required=False, default=None)
@click.option('-o', '--output_dir', type=click.Path(), required=False, default='')
def main(model_dir, iteration, scale, output_dir):
    model, config, model_name = load_model_checkp(dir=model_dir, iter=iteration, scale=scale)

    latentDim = model.config.categoryVectorDim_G
    device = 'cuda' if GPU_is_available() else 'cpu'

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)

    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = audioset_dict['values']
    as_att_idx = [labels.index(att) for att in audioset_keys]

    # n_gen = 50000
    _, val_feats = dummy_loader.get_validation_set(50000)

    n_gen = 10
    train_feats = torch.Tensor(dummy_loader.metadata)
    
    print(f"Numbre of instances = {n_gen}")

    all_feats = torch.cat([train_feats, val_feats], dim=0)
    all_audios = dummy_loader.data + dummy_loader.val_data


    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"fix_pf_rand_z")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))


    rand_index = torch.randint(len(all_audios), (n_gen,))
    labels_in = all_feats[rand_index]
    true_audios = np.array([all_audios[i] for i in rand_index])
    true_audios = torch.Tensor(list(map(postprocess, true_audios)))

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