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

    n_gen = min(80, len(dummy_loader.data))
    rand_idx = torch.randperm(len(dummy_loader.data))[:n_gen]
    train_feats = torch.Tensor(dummy_loader.metadata)
    
    print(f"Numbre of instances = {n_gen}")

    all_feats = torch.cat([train_feats[:, :-1], val_feats[:, :-1]], dim=0)
    cov = torch.FloatTensor(np.cov(all_feats, rowvar=False))
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0)

    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = audioset_dict['values']
    audioset_means = audioset_dict['mean']

    as_att_idx = [labels.index(att) for att in audioset_keys]

    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"dark_knowledge_eval")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    bsize = 80
    audioset_clf = AudioTagging()

    if results == '':
        results = pd.DataFrame(columns=['attribute', 'corr', 'increment', 'increment/std'])

    offset_list = [0] + list(np.logspace(-3, 0.6, 10))

    # for key in df.loc[(df['tr_corr'] > 0.1)]['attribute'].values:
    for key in audioset_keys[:10]:
        mean_in = mean
        att_idx = audioset_keys.index(key)
        for offset in offset_list:
            
            mean_in[att_idx] += std[att_idx]*offset
            mv_gaussian = torch.distributions.MultivariateNormal(loc=mean_in, covariance_matrix=cov)

            pred = []
            true = []
            pbar1 = trange(int(np.floor(n_gen/bsize)), desc=f'{key} loop')
            diff = []
            for i in pbar1:
                z, _ = model.buildNoiseData(1, skipAtts=True)
                pitch = train_feats[i*bsize:(i+1)*bsize, -1:]
                rand_feat_batch = torch.cat(
                    [mv_gaussian.sample((len(pitch),)), pitch], dim=1)
                
                orig_labels = train_feats[rand_idx[i*bsize:(i+1)*bsize]]
                labels_in = orig_labels.clone()
                # labels_in[:, audioset_keys.index(key)] = rand_feat_batch[:, audioset_keys.index(key)]
                labels_in[:, att_idx] += offset * std[att_idx]

                # sample frm G using rand labels
                z_rand, _ = model.buildNoiseData(labels_in.size(0), inputLabels=labels_in, skipAtts=True)
                # set the same noise vector z
                z_rand[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
                # sample from G
                rand_gen_batch = model.test(z_rand.to(device), toCPU=not GPU_is_available()).cpu()
                # postprocess output
                rand_gen_audio_batch = torch.Tensor(list(map(postprocess, rand_gen_batch)))
                # predict features with audioset classifier
                rbatch_pred = torch.Tensor(audioset_clf.inference(rand_gen_audio_batch)[0])
                # get those atts that have been used to train G
                rbatch_pred = rbatch_pred[:, as_att_idx]
                pred.append(rbatch_pred)
                # append the input conditioning features
                true.append(labels_in[:, :len(audioset_keys)])
                diff.append(abs(orig_labels[:, att_idx] - rbatch_pred[:, att_idx]))


            pred = torch.cat(pred, dim=0)
            true = torch.cat(true, dim=0)
            diff = torch.cat(diff, dim=0)
            

            ratt_in = true.transpose(1, 0)
            ratt_pred = pred.transpose(1, 0)

            in_att = ratt_in[att_idx]
            pred_att = ratt_pred[att_idx]
            orig_att = train_feats[rand_idx[:len(pred_att)], att_idx]


            rand_corr = np.corrcoef(in_att, pred_att)[0][1]
            # rand_cov = np.cov(ratt_in[i], ratt_pred[i])[0][1]
            results = results.append({
                'attribute': key,
                'std_offset': offset,
                'real_offset': std[att_idx]*offset,
                'pred_std': ((pred_att - orig_att) / std[att_idx]).mean(),
                'corr': rand_corr,
                'diff': abs(ratt_in[att_idx]-ratt_pred[att_idx]).mean(),
                'diff2': diff.mean()}, ignore_index=True)

            print(f"{key}: offset = {offset/std[att_idx]:.4f}; corr = {rand_corr:.4f}")
    torch.cuda.empty_cache()
    results.to_pickle(f'{output_dir}/out-of-dist_corr_{datetime.now().strftime("%H:%M:%S")}.pkl')
    with open(f'{output_dir}/out-of-dist_corr_{datetime.now().strftime("%H:%M:%S")}.txt', 'w') as file:
        file.write(results.to_latex(index=False))

    for k in np.unique(results['attribute'].values):
        plt.figure()
        ax = sns.lineplot(data=results.loc[results['attribute']==k], x="increment/std", y="corr")
        ax.set_title(k)
        ax.get_figure().savefig(f'{output_dir}/{k}_out-of-dist_corr_{datetime.now().strftime("%H:%M:%S")}.pdf')
        plt.close()



if __name__ == "__main__":
    main()