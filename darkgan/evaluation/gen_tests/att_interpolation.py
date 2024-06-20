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


def radial_interpolation(att_0: torch.Tensor, att_1: torch.Tensor, steps: int=10):
    z_batch = torch.zeros(steps, att_0.size(0))
    for k in range(steps):
        z_batch[k] = att_0 * (1 - k/steps) + att_1 * k/steps
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
    # loader_config["criteria"]["size"] = 1000
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)
    
    if os.path.exists(args.outdir):
        output_dir = args.outdir
    else:
        output_dir = args.dir
    output_dir = mkdir_in_path(output_dir, f"att_interpolation")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))

    for i in range(10):

        labels = torch.Tensor(random.sample(loader.metadata, k=args.n_gen))
        atts = radial_interpolation(labels[0, :128], labels[-1, :128])
        labels[:, :128] = atts
        labels[:, -1] = torch.randint(26, (1,))
        z, _ = model.buildNoiseData(args.n_gen, inputLabels=labels, skipAtts=True)
        z[:, :latentDim] = z[0, :latentDim]

        gnet = model.getOriginalG()
        gnet.eval()
        with torch.no_grad():
            out = gnet(z.to(device)).detach().cpu()

            audio_out = loader.postprocess(out)
            audio_out = torch.Tensor(list(map(lambda x: (x - x.mean())/(x.max() - x.mean()), audio_out)))
            audio_out = torch.cat([audio_out, torch.zeros((args.n_gen, 2048))], dim=1)
        # Create output evaluation dir
        saveAudioBatch(audio_out.flatten().reshape((1, -1)),
                       path=output_dir,
                       basename=f'att_interpolation_{i}', 
                       sr=config["transform_config"]["sample_rate"])
    print("FINISHED!\n")