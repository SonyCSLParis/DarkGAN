import os
import torch
import ipdb
import random

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch, get_device
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
from data.loaders import get_data_loader


def spherical_interpolation(z_dim: int, n_samples: int=10):
    x = torch.randn(z_dim)
    y = torch.randn(z_dim)
    cos_theta = torch.dot(x, y) / (x.norm() *  y.norm())
    theta = torch.acos(cos_theta)
    z_batch = torch.zeros(n_samples, z_dim)
    for k in range(n_samples):
        t = float(k/(n_samples - 1))
        z_batch[k] = (torch.sin((1-t)*theta)*x + torch.sin(t*theta)*y) / torch.sin(theta)
    return z_batch

def generate(parser):
    args = parser.parse_args()
    device = get_device()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.noiseVectorDim
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    # We load a dummy data loader for post-processing
    processor = AudioProcessor(**transform_config)

    dbname = loader_config['dbname']
    loader_config["criteria"]["size"] = 1000
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    # Create output evaluation dir
    if os.path.exists(args.outdir):
        output_dir = args.outdir
    else:
        output_dir = args.dir
    output_dir = mkdir_in_path(output_dir, f"spherical_interpolation")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))

    for i in range(10):
        label = torch.Tensor(random.sample(loader.metadata, k=1))

        labels, _ = model.buildNoiseData(1, inputLabels=label, skipAtts=True)
        z = labels.repeat(args.n_gen, 1)

        z_noise = spherical_interpolation(latentDim, args.n_gen)

        z[:, :latentDim] = z_noise

        gnet = model.getOriginalG()
        gnet.eval()
        with torch.no_grad():
            out = gnet(z.to(device)).detach().cpu()

            audio_out = loader.postprocess(out)
            audio_out = torch.cat([torch.Tensor(audio_out), torch.zeros((len(audio_out), 2048))], dim=1)

        saveAudioBatch(audio_out.flatten().reshape((1, -1)),
                       path=output_dir,
                       basename=f'test_spherical_interpolation_{i}', 
                       sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")