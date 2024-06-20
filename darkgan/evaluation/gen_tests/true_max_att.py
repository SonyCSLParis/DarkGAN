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

    _, val_feats = dummy_loader.get_validation_set(50000)

    train_feats = torch.Tensor(dummy_loader.metadata)
    


    all_feats = torch.cat([train_feats[:, :-1], val_feats[:, :-1]], dim=0)
    all_audios = dummy_loader.data + dummy_loader.val_data



    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"true_attribute_max")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))


    for key in audioset_keys:
        att_idx = audioset_keys.index(key)
        attribute_v = all_feats[:, att_idx]
        out_audios = [all_audios[k] for k in attribute_v.sort(descending=True)[1][:10]]
        # postprocess output
        audio_out = torch.Tensor(out_audios)
        audio_out = list(map(postprocess, audio_out))
        audio_out = torch.cat([torch.Tensor(audio_out), torch.zeros((len(out_audios), 2048))], dim=1)

        saveAudioBatch(audio_out.flatten().reshape((1, -1)),
               path=output_dir,
               basename=f'{key}_sample',
               sr=config["transform_config"]["sample_rate"])


if __name__ == "__main__":
    main()