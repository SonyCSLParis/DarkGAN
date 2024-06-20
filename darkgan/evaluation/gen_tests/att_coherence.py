import os

from datetime import datetime
from .generation_tests import *

from utils.utils import *
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
from gans.ac_criterion import ACGANCriterion
import numpy as np
import torch
import random
from data.loaders import get_data_loader
import ipdb
from tqdm import tqdm, trange
import pickle as pkl

def generate(parser):
    parser.add_argument("--val", dest="val", action='store_true')
    parser.add_argument("--train", dest="train", action='store_true')
    parser.add_argument("--avg-net", dest="avg_net", action='store_true')
    parser.add_argument("--name", dest="name", default="")
    parser.add_argument("--dump-labels", dest="dump_labels", action="store_true")
    parser.add_argument("--att", dest="attribute", type=str, required=True)
    parser.add_argument("--values", dest="values", nargs='+', type=float)
    parser.add_argument("--iter", dest="iteration", type=int)

    args = parser.parse_args()
    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    transform = transform_config['transform']
    loader_config = config['loader_config']
    # config['loader_config']['criteria']['size'] = 5
    processor = AudioProcessor(**transform_config)
    postprocess = processor.get_postprocessor()

    dbname = loader_config['dbname']
    loader_module = get_data_loader(dbname)
    dummy_loader = loader_module(name=dbname + '_' + transform, preprocessing=processor, **loader_config)
    labels = dummy_loader.get_validation_set(args.n_gen)[1]
    att_dict = dummy_loader.header['attributes']
    assert args.attribute in att_dict, f'Error: the selected attribute ({args.attribute}) \
        is not in the dictionary of the model ({att_dict.keys()})'
    att_classes = att_dict.keys()
    criterion = ACGANCriterion(att_dict)
    att_idx = criterion.keyOrder.index(args.attribute)
    att_size = criterion.attribSize[att_idx]
    start_idx = model.config.noiseVectorDim + sum(criterion.attribSize[:att_idx])

    if args.outdir == "":
        args.outdir = args.dir
    output_dir = mkdir_in_path(args.outdir, f"generation_samples")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "att_coherence")
    output_dir = mkdir_in_path(output_dir, args.attribute)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M'))

    assert len(labels) == args.n_gen, \
        f"Not enough validation data {len(labels)}. Decrease n-gen {args.n_gen}"

    if att_dict[args.attribute]['loss'] in ['bce', 'mse']:
        # att_vals = torch.Tensor([0.2, 0.5, 0.8])
        att_vals = torch.Tensor(args.values)
    elif att_dict[args.attribute]['loss'] == 'soft_xentropy':
        print("NOT IMPLMENTED YET")
        ipdb.set_trace()
    elif att_dict[args.attribute]['loss'] == 'xentropy':
        print("NOT IMPLMENTED YET")
        ipdb.set_trace()

    atts = att_dict[args.attribute]['values']

    device = 'cuda' if GPU_is_available() else 'cpu'
    with torch.no_grad():
        with open(f"{output_dir}/params_in.txt", "a") as f:
            pbar1 = trange(args.n_gen, desc="latent z loop")
            for i in pbar1:
                pbar2 = trange(len(atts), desc='Attribute loop')
                for j in pbar2:
                    pbar2.set_description(f"Generating for attribute: {atts[j]}")
                    z, _ = model.buildNoiseData(1,
                        inputLabels=labels[i].unsqueeze(0), skipAtts=True)
                    z = z.repeat(len(att_vals), 1)
                    z[:, start_idx + j] = att_vals

                    data_batch = model.test(z.to(device), toCPU=not GPU_is_available(), getAvG=args.avg_net).cpu()
                    # data_batch = torch.cat(data_batch, dim=0)
                    audio_out = map(postprocess, data_batch)

                    att_output_dir = mkdir_in_path(output_dir, atts[j])
                    saveAudioBatch(audio_out,
                                   path=att_output_dir,
                                   basename=f'z{i}_{atts[j]}_sample',
                                   sr=config["transform_config"]["sample_rate"])

                    for k in range(len(data_batch)):
                        pkl.dump(z[k, model.config.noiseVectorDim:].tolist(), 
                                open(os.path.join(att_output_dir, f'z{i}_{atts[j]}_sample_{k}.pkl'), 'wb'))

    print("FINISHED!\n")