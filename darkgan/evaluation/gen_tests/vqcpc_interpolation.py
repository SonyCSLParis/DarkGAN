import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
import numpy as np
import torch
import random
from data.loaders import get_data_loader
import ipdb
from tqdm import tqdm

from .spherical_interpolation import spherical_interpolation
from .radial_interpolation import radial_interpolation


def generate(parser):
    parser.add_argument("--val", dest="val", action='store_true')
    parser.add_argument("--spherical", dest="spherical", action='store_true')
    parser.add_argument("--single-file", dest="single", action='store_true')
    parser.add_argument("--train", dest="train", action='store_true')
    parser.add_argument("--avg-net", dest="avg_net", action='store_true')
    parser.add_argument("--name", dest="name", default="")
    parser.add_argument("--duration", dest="dur", default=1, type=float)

    args = parser.parse_args()
    args.n_gen = 10

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']

    processor = AudioProcessor(**transform_config)
    # processor.audio_length = int(args.dur * 16000)
    postprocess = processor.get_postprocessor()
    # transform_config['n_frames'] = 64
    processor2 = AudioProcessor(**transform_config)
    processor2.audio_length = int(args.dur * 16000)
    postprocess2 = processor2.get_postprocessor()

    # Create output evaluation dir
    if args.spherical:
        name = args.name + '_spherical'
    else:
        name = args.name + '_radial'

    if args.outdir == "":
        args.outdir = args.dir
    output_dir = mkdir_in_path(args.outdir, f"generation_samples")
    output_dir = mkdir_in_path(output_dir, f"interpolation")
    output_dir = mkdir_in_path(output_dir, model_name)

    output_dir = mkdir_in_path(output_dir, name + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M'))

    # loader_config["criteria"]["filter"]["seq_len"] = 64
    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    # labels = None
    # if model.config.ac_gan:
    #     if args.val:
    #         val_set = loader.get_validation_set()[1]
    #         perm = torch.randperm(val_set.size(0))
    #         idx = perm[:args.n_gen]
    #         labels = val_set[idx]
    #     else:
    for j in range(10):
        labels = torch.Tensor(random.sample(loader.metadata, k=args.n_gen))
        labels = labels[0:1].repeat(10, 1)

        dur_frames = int(args.dur * 16000 / 512)
        ups = torch.nn.Upsample(size=(1, dur_frames), mode='nearest')
        cpcl = ups(labels[:, None, None, 2:])
        labels = torch.cat([labels[:, :2], cpcl.reshape(-1, cpcl.size(-1))], dim=1)
        
        labels[:, 1] =  torch.randint(low=0, high=26, size=(1,))

        # labels = torch.cat([labels[:, :2], cpcl.reshape(-1, cpcl.size(-1))], dim=1)
        z, _ = model.buildNoiseData(args.n_gen, inputLabels=labels, skipAtts=True, test=True)
        
        interpolation = spherical_interpolation if args.spherical else radial_interpolation
        z.transpose(1, 3).transpose(0,2)[:, :, :, :model.config.noiseVectorDim] = interpolation(model.config.noiseVectorDim, args.n_gen)
     
        # z[:, :model.config.noiseVectorDim] = torch.randn(z[0, :model.config.noiseVectorDim].shape)
        gnet = model.netG.eval()

        data_batch = []
        with torch.no_grad():
            for i in range(int(np.ceil(args.n_gen/args.batch_size))):
                data_batch.append(
                    model.test(z[i*args.batch_size:args.batch_size*(i+1)],
                    toCPU=True, getAvG=args.avg_net).cpu())

            data_batch = torch.cat(data_batch, dim=0)
            audio_out = map(postprocess2, data_batch)
            audio_out = map(lambda x: x/x.max(), audio_out)
            if args.single:
                audio_out = map(lambda x: np.append(x, np.zeros(2048)), audio_out)
                audio_out = np.concatenate(list(audio_out))
                audio_out = audio_out[None, :]

        saveAudioBatch(audio_out,
                       path=output_dir,
                       basename=f'sample_{j}',
                       sr=config["transform_config"]["sample_rate"])
        print(f"Audios saved in {output_dir}")


    print("FINISHED!\n")