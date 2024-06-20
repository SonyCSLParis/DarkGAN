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
@click.option('--corr-table', type=click.Path(exists=True), required=True, default='')
def main(model_dir, iteration, scale, output_dir, corr_table):
    model, config, model_name = load_model_checkp(dir=model_dir, iter=iteration, scale=scale)
    df = pd.read_pickle(corr_table)
    ipdb.set_trace()
    latentDim = model.config.categoryVectorDim_G
    device = 'cuda' if GPU_is_available() else 'cpu'

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']
    loader_config['criteria']['audioset']['temperature'] = 1

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)
    

    _, val_feats = dummy_loader.get_validation_set(50000)

    n_gen = min(100, len(dummy_loader.data))
    rand_idx = torch.randperm(len(dummy_loader.data))[:n_gen]
    train_feats = torch.Tensor(dummy_loader.metadata)
    
    print(f"Numbre of instances = {n_gen}")

    all_feats = torch.cat([train_feats[:, :-1], val_feats[:, :-1]], dim=0)

    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = list(df['attribute'].values)

    # audioset_keys = audioset_dict['values']

    as_att_idx = [labels.index(att) for att in audioset_keys]

    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"violin_plot")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    bsize = 50
    audioset_clf = AudioTagging()

    results = pd.DataFrame(columns=['attribute'])

    for key in audioset_keys:
    # for key in df.loc[(df['tr_corr'] > 0.)]['attribute'].values:
        att_idx = audioset_keys.index(key)
        pred = []
        true = []
        pbar1 = trange(int(np.floor(n_gen/bsize)), desc=f'{key} loop')
        for i in pbar1:
            # z, _ = model.buildNoiseData(1, skipAtts=True)
            orig_labels = torch.stack(random.choices(train_feats, k=bsize))
            labels_in = orig_labels.clone()

            offset = torch.rand(labels_in.size(0))*2 + 2 
            labels_in[:, att_idx] = offset
            z_rand, _ = model.buildNoiseData(labels_in.size(0), inputLabels=labels_in, skipAtts=True)
            # z_rand[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
            y_increased_label = model.test(z_rand.to(device), toCPU=not GPU_is_available()).cpu()
            # postprocess output
            y_increased_label = torch.Tensor(list(map(postprocess, y_increased_label)))
            # predict features with audioset classifier
            pred_incremented = torch.Tensor(audioset_clf.inference(y_increased_label)[0])
            # get those atts that have been used to train G
            pred = pred_incremented[:, as_att_idx]

            labels_in[:, att_idx] = 0
            z_rand2, _ = model.buildNoiseData(labels_in.size(0), inputLabels=labels_in, skipAtts=True)
            z_rand2[:, :model.config.noiseVectorDim] = z_rand[:, :model.config.noiseVectorDim]
            # z_rand[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
            y_increased_label = model.test(z_rand2.to(device), toCPU=not GPU_is_available()).cpu()
            # postprocess output
            y_increased_label = torch.Tensor(list(map(postprocess, y_increased_label)))
            # predict features with audioset classifier
            true = torch.Tensor(audioset_clf.inference(y_increased_label)[0])
            # get those atts that have been used to train G
            true = true[:, as_att_idx]

            pred = pred.transpose(1, 0)[att_idx]
            true = true.transpose(1, 0)[att_idx]
            for p, t, o in zip(pred, true, offset):
                results = results.append({
                    'attribute': key,
                    'x': o.numpy(),
                    'y': p.numpy(),
                    'y1': (p - t).numpy()}, ignore_index=True)

            print(f"{key}: avg offset = {offset.mean().numpy()}, avg pred = {(pred - true).mean().numpy():.4f}")
    torch.cuda.empty_cache()
    results.to_pickle(f'{output_dir}/pred_feat1.pkl')

if __name__ == "__main__":
    main()