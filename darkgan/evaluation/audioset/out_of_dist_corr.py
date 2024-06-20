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
@click.option('--results', type=click.Path(), required=False, default='')
# @click.option('-t', '--test', type=str, required=True, default='range')
def main(model_dir, iteration, scale, output_dir, corr_table, results):
    model, config, model_name = load_model_checkp(dir=model_dir, iter=iteration, scale=scale)
    df = pd.read_pickle(corr_table)

    latentDim = model.config.categoryVectorDim_G
    device = 'cuda' if GPU_is_available() else 'cpu'

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']
    # loader_config['criteria']['audioset']['temperature'] = 1

    # Comment when not testing!!!
    
    # loader_config["criteria"]["size"] = 5000

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)
    
    # n_gen = 50000
    _, val_feats = dummy_loader.get_validation_set(50000)

    n_gen = min(100, len(dummy_loader.data))
    rand_idx = torch.randperm(len(dummy_loader.data))[:n_gen]
    train_feats = torch.Tensor(dummy_loader.metadata)
    
    print(f"Numbre of instances = {n_gen}")

    all_feats = torch.cat([train_feats[:, :-1], val_feats[:, :-1]], dim=0)
    cov = torch.FloatTensor(np.cov(all_feats, rowvar=False))
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0)

    audioset_dict = dummy_loader.header['attributes']['audioset']
    # 
    # audioset_keys = audioset_dict['values']
    audioset_keys = list(df['attribute'].values)
    audioset_means = audioset_dict['mean']

    as_att_idx = [labels.index(att) for att in audioset_keys]

    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"dark_knowledge_eval_latest2")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    bsize = 10
    audioset_clf = AudioTagging()

    if results == '':
        results = pd.DataFrame(columns=['attribute', 'corr', 'increment', 'increment/std'])

    offsets = np.arange(0, 5, 0.2)
    for key in df.sort_values(by='tr_corr', ascending=False)['attribute'][:50].values:
    # for key in df.loc[(df['tr_corr'] > 0.)]['attribute'].values:
    # for key in audioset_keys:
        att_idx = audioset_keys.index(key)
        for offset in offsets:
            pred = []
            true = []
            pbar1 = trange(int(np.floor(n_gen/bsize)), desc=f'{key} loop')
            for i in pbar1:
                z, _ = model.buildNoiseData(1, skipAtts=True)
                orig_labels = torch.stack(random.choices(train_feats, k=bsize))
                labels_in = orig_labels.clone()
                # labels_in[:, att_idx] = offset
                # labels_in[:, att_idx] += offset * std[att_idx]
                labels_in[:, att_idx] += offset

                # sample frm G using rand labels
                z_orig, _ = model.buildNoiseData(labels_in.size(0), inputLabels=orig_labels, skipAtts=True)
                z_orig[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
                y_orig_label = model.test(z_orig.to(device), toCPU=not GPU_is_available()).cpu()
                
                z_rand, _ = model.buildNoiseData(labels_in.size(0), inputLabels=labels_in, skipAtts=True)
                z_rand[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
                y_increased_label = model.test(z_rand.to(device), toCPU=not GPU_is_available()).cpu()

                # postprocess output
                y_orig_label = torch.Tensor(list(map(postprocess, y_orig_label)))
                y_increased_label = torch.Tensor(list(map(postprocess, y_increased_label)))
                # predict features with audioset classifier
                pred_orig = torch.Tensor(audioset_clf.inference(y_orig_label)[0])
                pred_incremented = torch.Tensor(audioset_clf.inference(y_increased_label)[0])
                
                # get those atts that have been used to train G
                pred_orig = pred_orig[:, as_att_idx]
                pred_incremented = pred_incremented[:, as_att_idx]
                pred.append(pred_incremented)
                true.append(pred_orig)

            pred = torch.cat(pred, dim=0).transpose(1, 0)[att_idx]
            true = torch.cat(true, dim=0).transpose(1, 0)[att_idx]

            results = results.append({
                'attribute': key,
                'offset': offset,
                'offset/std': offset/true.std().numpy(),
                # 'std_offset': offset,
                # 'real_offset': offset * std[att_idx],
                'p-t/std': (pred - true).mean().numpy()/true.std().numpy(),
                'p-t/std_all': (pred - true).mean().numpy()/std[att_idx].numpy(),
                'p-t': (pred - true).mean().numpy()}, ignore_index=True)

            print(f"{key}: offset = {offset:.4f}; p-t/std = {(pred - true).mean().numpy()/true.std().numpy():.4f}")
    torch.cuda.empty_cache()
    results.to_pickle(f'{output_dir}/out-of-dist_corr_{datetime.now().strftime("%H:%M:%S")}.pkl')
    with open(f'{output_dir}/out-of-dist_corr.txt', 'w') as file:
        file.write(results.to_latex(index=False))

    # for k in np.unique(results['attribute'].values):
    #     plt.figure()
    #     ax = sns.lineplot(data=results.loc[results['attribute']==k], x="increment/std", y="corr")
    #     ax.set_title(k)
    #     ax.get_figure().savefig(f'{output_dir}/{k}_out-of-dist_corr_{datetime.now().strftime("%H:%M:%S")}.pdf')
    #     plt.close()



if __name__ == "__main__":
    main()