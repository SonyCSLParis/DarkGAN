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
    # loader_config["criteria"]["size"] = 1000

    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)

    

    feats = torch.Tensor(dummy_loader.metadata)
    val_feats = torch.Tensor(dummy_loader.val_labels)
    all_feats = torch.cat([feats, val_feats])

    audioset_feats = feats[:, :-1]
    audioset_dict = dummy_loader.header['attributes']['audioset']
    audioset_keys = audioset_dict['values']
    audioset_means = audioset_dict['mean']

    if output_dir == "":
        output_dir = model_dir
    output_dir = mkdir_in_path(output_dir, f"hist_feat_coherence")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    n_gen = 200
    n_bins = 16
    bsize = 16
    audioset_clf = AudioTagging(
        checkpoint_path=f"/home/javier/panns_data/Cnn14_16k_mAP=0.438.pth",
        device=device)
    # pbar1 = trange(5, desc='Attribute loop')
    pbar1 = trange(len(audioset_keys), desc='Attribute loop')

    _j = [0, 0.5, 1, 5]
    import pandas as pd
    violin_plot = pd.DataFrame(columns=['att', 'diff', 'val'])
    violin_plot2 = pd.DataFrame(columns=['att', 'diff', 'val'])
    for i in pbar1:
        as_key = audioset_keys[i]
        att_idx = labels.index(as_key)

        pbar1.set_description(as_key)

        rand_idx = torch.randperm(audioset_feats.size(0))[:n_gen]
        real_data = torch.Tensor([dummy_loader.data[k] for k in rand_idx])
        real_audio = torch.Tensor(list(map(postprocess, real_data)))
        
        
        real_feats = audioset_feats[rand_idx, i]
        
        feat_mean = audioset_means[as_key]

        att_offset = model.config.noiseVectorDim


        fakes = []
       
        
        for j in _j:
            
            input_feats = feats[rand_idx]

            z, _ = model.buildNoiseData(n_gen, inputLabels=input_feats, skipAtts=True)
            # z[:, att_offset + i] = z[:, att_offset + i]*j
            z[:, att_offset + i] += j
            pred = []
            real_preds = []
            for b in trange(int(np.ceil(len(z)/bsize))):
                real_preds.append(torch.Tensor(audioset_clf.inference(real_audio[b*bsize:(b+1)*bsize])[0])[:, att_idx])
                data_batch = model.test(z[b*bsize:(b+1)*bsize].to(device), toCPU=not GPU_is_available()).cpu()
                audio_out = torch.Tensor(list(map(postprocess, data_batch)))
                batc_pred = torch.Tensor(audioset_clf.inference(audio_out)[0])
                pred.append(batc_pred[:, att_idx])
            

            real_preds = torch.cat(real_preds, dim=0)
            pred = torch.cat(pred, dim=0)
            val = (pred - input_feats[:, i])/input_feats[:, i]
            for v in val:
                violin_plot = violin_plot.append({
                    'att': audioset_keys[i], 
                    'diff': j,
                    'val': np.float(v)},
                    ignore_index=True)

            violin_plot2 = violin_plot2.append({
                'att': audioset_keys[i], 
                'diff': j,
                'val': np.float((pred.mean() - real_preds.mean())/real_preds.mean())},
                ignore_index=True)

            fakes.append(pred)

        print(f"Real feats mean: {real_feats.mean():.4f}")
        print(f"Real preds mean: {real_preds.mean():.4f}")
        print(f"Gen +0 mean: {fakes[0].mean():.4f}")
        print(f"Gen +0.5 mean: {fakes[1].mean():.4f}")
        print(f"Gen +1 mean: {fakes[2].mean():.4f}")
        print(f"Gen +10 mean: {fakes[3].mean():.4f}")

        # _min = min(real_preds.mean() - real_preds.std()/2, fakes[0].mean() - fakes[0].std()/2, fakes[1].mean() - fakes[1].std()/2, fakes[2].mean() - fakes[2].std()/2, fakes[3].mean() - fakes[3].std()/2)
        # # _min = min(real_feats.mean() - real_feats.std()/2, real_preds.mean() - real_preds.std()/2, fakes[0].mean() - fakes[0].std()/2, fakes[1].mean() - fakes[1].std()/2, fakes[2].mean() - fakes[2].std()/2, fakes[3].mean() - fakes[3].std()/2)
        # _min = max(0, _min)
        # _max = max(real_preds.mean() + real_preds.std()/2, fakes[0].mean() + fakes[0].std()/2, fakes[1].mean() + fakes[1].std()/2, fakes[2].mean() + fakes[2].std()/2, fakes[3].mean() + fakes[3].std()/2)
        # # _max = max(real_feats.mean() + real_feats.std()/2, real_preds.mean() + real_preds.std()/2, fakes[0].mean() + fakes[0].std()/2, fakes[1].mean() + fakes[1].std()/2, fakes[2].mean() + fakes[2].std()/2, fakes[3].mean() + fakes[3].std()/2)
        # _max = min(1, _max)

        # ipdb.set_trace()
        # real_dist = torch.histc(real_feats, bins=n_bins, min=_min, max=_max)
        # real_dist2 = torch.histc(real_preds, bins=n_bins, min=_min, max=_max)
        # fake_dist_x1 = torch.histc(fakes[0], bins=n_bins, min=_min, max=_max)
        # fake_dist_x10 = torch.histc(fakes[1], bins=n_bins, min=_min, max=_max)
        # fake_dist_x100 = torch.histc(fakes[2], bins=n_bins, min=_min, max=_max)
        # fake_dist_x1000 = torch.histc(fakes[3], bins=n_bins, min=_min, max=_max)

        # from matplotlib import pyplot as plt
        import plotly.graph_objects as go
        from plotly.offline import plot
        from plotly.figure_factory import create_distplot
        from scipy.stats import wasserstein_distance
        import matplotlib.pyplot as plt
        # x = np.arange(_min, _max, (_max - _min)/n_bins)

        hist_data = [real_preds] + fakes
        # hist_data = [h / _max for h in hist_data]
        hist_data = [h.numpy() for h in hist_data]
        group_labels = ['real'] + [str(x) for x in _j]
        # hist_data = [list(real_preds), list(fakes[0]), list(fakes[1]), list(fakes[2]), list(fakes[3])]
        # group_labels = ['real', '+0', '+0.5', '+1', '+10']
        fig = create_distplot(
            hist_data, group_labels, 
            bin_size=.002,
            curve_type='normal')


        fig.update_layout(title_text=f'{as_key} value distribution')
        fig.update_layout(barmode='group')
        fig.show()
        plot(fig, filename=f'{output_dir}/{as_key}.html')

        import seaborn as sns
        import pandas as pd
        df = pd.DataFrame({k: x for k, x in zip(group_labels, hist_data)})
        fig = sns.displot(
            df,
            kind="kde", 
            fill=True,
            # multiple="stack", 
            bw_adjust=.25)
        fig.savefig(f'{output_dir}/{as_key}.pdf')
        # plot(fig, filename=f'{output_dir}/{as_key}.png')
        plt.close()

    plt.close()
    plt.figure("TEST")
    ax = sns.violinplot(
        y='val', x='diff', scale='width', data=violin_plot)
    ax.set_yscale("log")
    ax.get_figure().savefig(f'{output_dir}/violin_plot.pdf')

    plt.close()
    plt.figure("TEST2")
    ax2= sns.violinplot(
        y='val', x='diff', scale='width', data=violin_plot2)
    ax2.set_yscale("log")
    ax2.get_figure().savefig(f'{output_dir}/violin_plot2.pdf')
    print("FINISHED!\n")

if __name__ == '__main__':
    main()