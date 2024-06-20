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


def generate(parser):
    parser.add_argument("--val", dest="val", action='store_true')
    parser.add_argument("--single-file", dest="single", action='store_true')
    parser.add_argument("--train", dest="train", action='store_true')
    parser.add_argument("--avg-net", dest="avg_net", action='store_true')
    parser.add_argument("--name", dest="name", default="")
    parser.add_argument("--duration", dest="dur", default=1, type=float)

    args = parser.parse_args()
    args.n_gen = 26

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']

    processor = AudioProcessor(**transform_config)

    # transform_config['n_frames'] = 64
    processor2 = AudioProcessor(**transform_config)
    processor2.audio_length = int(args.dur * 16000)
    postprocess2 = processor2.get_postprocessor()

    # Create output evaluation dir
    if args.val:
        name = args.name + '_val_labels'
    elif args.train:
        name = args.name + '_train_labels'
    else:
        name = args.name + '_rand_labels'
    if args.outdir == "":
        args.outdir = args.dir
    output_dir = mkdir_in_path(args.outdir, f"generation_samples")
    output_dir = mkdir_in_path(output_dir, f"scales")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, f"dur_{args.dur}")
    output_dir = mkdir_in_path(output_dir, name + '_' + datetime.now().strftime('%Y-%m-%d_%H_%M'))

    # loader_config["criteria"]["filter"]["seq_len"] = 64
    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    labels = None


    for j in range(10):
        if model.config.ac_gan:
            if args.val:
                # val_set = loader.get_validation_set()[1]
                labels = torch.Tensor(random.sample(loader.val_labels, k=args.n_gen))
                # perm = torch.randperm(val_set.size(0))
                # idx = perm[:args.n_gen]
                # labels = val_set[idx]
            else:
                labels = torch.Tensor(random.sample(loader.metadata, k=args.n_gen))

        ups = torch.nn.Upsample(scale_factor=(1, args.dur), mode='nearest')


        cpcl = ups(labels[:, None, None, 2:])
        cpcl = cpcl[0:1].repeat(cpcl.size(0), 1, 1, 1)
        labels[:, 1] = torch.Tensor(list(range(0, len(labels))))
        labels = torch.cat([labels[:, :2], cpcl.reshape(-1, cpcl.size(-1))], dim=1)

        # z.transpose(1, 3)[:, :, :, :model.config.noiseVectorDim] = torch.randn(model.config.noiseVectorDim)
     
        # z[:, :model.config.noiseVectorDim] = torch.randn(z[0, :model.config.noiseVectorDim].shape)
        gnet = model.netG.eval()
        z, _ = model.buildNoiseData(args.n_gen, inputLabels=labels, skipAtts=True, test=True)
        z[:, :128, :, :] = torch.randn(1, model.config.noiseVectorDim, 1, 1).repeat(1, 1, 1, z.size(-1))

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



        # audio_out = map(lambda x: (x-x.mean())/(x.max()-x.mean()), audio_out)

        saveAudioBatch(audio_out,
                       path=output_dir,
                       basename=f'sample_{j}',
                       sr=config["transform_config"]["sample_rate"])
        print(f"Audios saved in {output_dir}")
    # if args.dump_labels:
    #     with open(f"{output_dir}/params_in.txt", "a") as f:
    #         for i in tqdm(range(args.n_gen), desc='Creating Samples'):
    #             params = labels[i, :-1].tolist()
    #             f.writelines([f"{i}, {list(params)}\n"])

    print("FINISHED!\n")