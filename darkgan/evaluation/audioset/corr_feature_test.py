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


@click.command()
@click.option('-d', '--model_dir', type=click.Path(exists=True), required=True, default='')
@click.option('--iteration', type=int, required=False, default=None)
@click.option('--scale', type=int, required=False, default=None)
@click.option('-o', '--output_dir', type=click.Path(), required=False, default='')
# @click.option('-t', '--test', type=str, required=True, default='range')
# @click.option('-t', '--test', type=str, required=True, default='range')
def main(model_dir, iteration, scale, output_dir):
    model, config, model_name = load_model_checkp(dir=model_dir, iter=iteration, scale=scale)
    latentDim = model.config.categoryVectorDim_G
    device = 'cuda' if GPU_is_available() else 'cpu'

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']

    # Comment when not testing!!!
    # n_gen = 200
    # loader_config["criteria"]["size"] = 5000

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)
    

    
    n_gen = 50000
    _, val_feats = dummy_loader.get_validation_set(50000)
    n_gen = min(n_gen, len(val_feats))
    rand_idx = torch.randperm(len(dummy_loader.data))[:n_gen]
    train_feats = torch.Tensor(dummy_loader.metadata)
    # train_audio = torch.Tensor(dummy_loader.data)[rand_idx]
    mv_gaussian = torch.distributions.MultivariateNormal(loc=train_feats[:, :-1].mean(dim=0), covariance_matrix=torch.FloatTensor(np.cov(train_feats[:, :-1], rowvar=False)))

    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = audioset_dict['values']
    audioset_means = audioset_dict['mean']

    as_att_idx = [labels.index(att) for att in audioset_keys]

    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"corr_feat_coherence")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    bsize = 50
    audioset_clf = AudioTagging()

    real_tpred = []
    real_vpred = []
    real_rpred = []
    tpred = []
    vpred = []
    rpred = []
    
    pbar1 = trange(int(np.floor(val_feats.size(0)/bsize)), desc='Generation loop')
    for i in pbar1:
        z, _ = model.buildNoiseData(1, skipAtts=True)
        tr_feat_batch = train_feats[rand_idx][i*bsize:(i+1)*bsize]
        val_feat_batch = val_feats[i*bsize:(i+1)*bsize]
        rand_feat_batch = torch.cat(
            [mv_gaussian.sample((len(val_feat_batch),)), 
            # add pitch
            val_feat_batch[:, -1].unsqueeze(-1)], dim=1)


        # tr_audio_batch = train_audio[i*bsize:(i+1)*bsize]
        # val_audio_batch = val_audio[i*bsize:(i+1)*bsize]

        # sample from G using train labels
        z_tr, _ = model.buildNoiseData(tr_feat_batch.size(0), inputLabels=tr_feat_batch, skipAtts=True)
        # set the same noise vector z
        z_tr[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
        # sample frm G using val labels
        z_val, _ = model.buildNoiseData(val_feat_batch.size(0), inputLabels=val_feat_batch, skipAtts=True)
        # set the same noise vector z
        z_val[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
        # sample frm G using rand labels
        z_rand, _ = model.buildNoiseData(rand_feat_batch.size(0), inputLabels=rand_feat_batch, skipAtts=True)
        # set the same noise vector z
        z_rand[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]
        

        # sample from G
        tr_gen_batch = model.test(z_tr.to(device), toCPU=not GPU_is_available()).cpu()
        val_gen_batch = model.test(z_val.to(device), toCPU=not GPU_is_available()).cpu()
        rand_gen_batch = model.test(z_rand.to(device), toCPU=not GPU_is_available()).cpu()
        
        # postprocess output
        tr_gen_audio_batch = torch.Tensor(list(map(postprocess, tr_gen_batch)))
        val_gen_audio_batch = torch.Tensor(list(map(postprocess, val_gen_batch)))
        rand_gen_audio_batch = torch.Tensor(list(map(postprocess, rand_gen_batch)))
        
        # predict features with audioset classifier
        tbatch_pred = torch.Tensor(audioset_clf.inference(tr_gen_audio_batch)[0])
        vbatch_pred = torch.Tensor(audioset_clf.inference(val_gen_audio_batch)[0])
        rbatch_pred = torch.Tensor(audioset_clf.inference(rand_gen_audio_batch)[0])
        # get those atts that have been used to train G
        tbatch_pred = tbatch_pred[:, as_att_idx]
        vbatch_pred = vbatch_pred[:, as_att_idx]
        rbatch_pred = rbatch_pred[:, as_att_idx]
        tpred.append(tbatch_pred)
        vpred.append(vbatch_pred)
        rpred.append(rbatch_pred)

        # Do the same for real audio
        # tr_audio_batch = torch.Tensor(list(map(postprocess, tr_audio_batch)))
        # val_audio_batch = torch.Tensor(list(map(postprocess, val_audio_batch)))
        # real_tbatch_pred = torch.Tensor(audioset_clf.inference(tr_audio_batch)[0])
        # real_vbatch_pred = torch.Tensor(audioset_clf.inference(val_audio_batch)[0])
        # real_tbatch_pred = real_tbatch_pred[:, as_att_idx]
        # real_vbatch_pred = real_vbatch_pred[:, as_att_idx]
        # real_tpred.append(real_tbatch_pred)
        # real_vpred.append(real_vbatch_pred)

        # append the input conditioning features
        real_tpred.append(tr_feat_batch[:, :len(audioset_keys)])
        real_vpred.append(val_feat_batch[:, :len(audioset_keys)])
        real_rpred.append(rand_feat_batch[:, :len(audioset_keys)])


    tpred = torch.cat(tpred, dim=0)
    vpred = torch.cat(vpred, dim=0)
    rpred = torch.cat(rpred, dim=0)
    real_tpred = torch.cat(real_tpred, dim=0)
    real_vpred = torch.cat(real_vpred, dim=0)
    real_rpred = torch.cat(real_rpred, dim=0)
    
    # remove pitch
    val_feats = val_feats[:, :-1]
    tr_feats = train_feats[:, :-1]
    result = {}

    # tatt_in = tr_feats.transpose(1, 0)
    # vatt_in = val_feats.transpose(1, 0)    
    tatt_in = real_tpred.transpose(1, 0)
    vatt_in = real_vpred.transpose(1, 0)
    ratt_in = real_rpred.transpose(1, 0)
    tatt_pred = tpred.transpose(1, 0)
    vatt_pred = vpred.transpose(1, 0)
    ratt_pred = rpred.transpose(1, 0)

    val_all_corr = []
    tr_all_corr = []
    rand_all_corr = []

    df = pd.DataFrame(columns=['attribute', 'tr_corr', 'val_corr', 'rand_corr'])
    for i, att in enumerate(audioset_keys):
        key = audioset_keys[i]
        tr_corr = np.corrcoef(tatt_in[i], tatt_pred[i])[0][1]
        val_corr = np.corrcoef(vatt_in[i], vatt_pred[i])[0][1]
        rand_corr = np.corrcoef(ratt_in[i], ratt_pred[i])[0][1]
        
        val_all_corr.append(val_corr)
        tr_all_corr.append(tr_corr)
        rand_all_corr.append(rand_corr)
        
        tr_cov = np.cov(tatt_in[i], tatt_pred[i])[0][1]
        val_cov = np.cov(vatt_in[i], vatt_pred[i])[0][1]
        rand_cov = np.cov(ratt_in[i], ratt_pred[i])[0][1]

        df = df.append({
            'attribute': key, 
            'tr_corr': tr_corr, 
            'val_corr': val_corr, 
            'rand_corr': rand_corr}, ignore_index=True)

        print(f"{att}: tr_corr = {tr_corr:.4f}; tr_cov = {tr_cov:.4f}")
        print(f"{att}: val_corr = {val_corr:.4f}; val_cov = {val_cov:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tatt_in[i], y=tatt_pred[i],
                    mode='markers',
                    name='markers'))
        fig.add_trace(go.Scatter(x=[0.0, max(tatt_in[i])], y=[0.0, tr_corr*max(tatt_in[i])],
                    mode='lines',
                    name='correlation'))
        
        fig.update_layout(title=att, xaxis_title="real", yaxis_title="pred")
        plot(fig, filename=f'{output_dir}/evaluation_corr_{att}_{datetime.now().strftime("%H:%M:%S")}.html')
        plt.close()

    with open(f'{output_dir}/corr_coherence_{datetime.now().strftime("%H:%M:%S")}.txt', 'w') as file:
        file.write(df.to_latex(index=False))

    df.to_pickle(f'{output_dir}/evaluation_corr_coherence_{datetime.now().strftime("%H:%M:%S")}.pkl')
    tr_order_idx = np.argsort(tr_all_corr)[::-1]
    tr_all_corr.sort(reverse=True)
    val_order_idx = np.argsort(val_all_corr)[::-1]
    val_all_corr.sort(reverse=True)
    rand_order_idx = np.argsort(rand_all_corr)[::-1]
    rand_all_corr.sort(reverse=True)

    print(f"Val att order:\n{list(zip(np.array(audioset_keys)[val_order_idx[:10]], val_all_corr[:10]))}")
    print(f"Train att order: \n{list(zip(np.array(audioset_keys)[tr_order_idx[:10]], tr_all_corr[:10]))}")
    print(f"Train att order: \n{list(zip(np.array(audioset_keys)[rand_order_idx[:10]], rand_all_corr[:10]))}")

    save_json(result, f'{output_dir}/evaluation_corr_coherence_{datetime.now().strftime("%H:%M:%S")}.json')

if __name__ == "__main__":
    main()