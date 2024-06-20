import os
import random

from datetime import datetime
from .generation_tests import *
from data.loaders import get_data_loader

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor

import ipdb

def generate(parser):
    parser.add_argument("--single-file", dest="single", action='store_true')
    parser.add_argument("--val", dest="val", action='store_true')
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    assert 'pitch' in model.ClassificationCriterion.keyOrder, "Model has no pitch conditioning"
    pitch_idx = model.ClassificationCriterion.keyOrder.index('pitch')
    args.n_gen = model.ClassificationCriterion.attribSize[pitch_idx]
    att_offset = sum(model.ClassificationCriterion.attribSize[:pitch_idx])
    # We load a dummy data loader for post-processing
    processor = AudioProcessor(**config['transform_config'])
    postprocess = processor.get_postprocessor()
    transform_config = config['transform_config']
    loader_config = config['loader_config']
    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    # Create output evaluation dir
    if os.path.exists(args.outdir):
        output_dir = args.outdir
    else:
        output_dir = args.dir
    output_dir = mkdir_in_path(output_dir, f"pitch_scale")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    # Create evaluation manager
    # eval_manager = StyleGEvaluationManager(model, n_gen=100)
    output_path = mkdir_in_path(output_dir, f"one_z_pitch_sweep")

    for i in range(10):
        if args.val:
            label = torch.Tensor(random.sample(loader.val_labels, k=1))
        else:
            label = torch.Tensor(random.sample(loader.metadata, k=1))

        labels = label.repeat(args.n_gen, 1)
        labels[:, att_offset:att_offset + args.n_gen] = torch.from_numpy(np.arange(args.n_gen)).unsqueeze(1)
        z, _ = model.buildNoiseData(args.n_gen, inputLabels=labels, skipAtts=True)
        z[:, :model.config.noiseVectorDim] = z[0, :model.config.noiseVectorDim]

        gen_batch = model.test(z, toCPU=True).cpu()
        audio_out = map(postprocess, gen_batch)
        audio_out = list(map(lambda x: (x - x.mean())/(x.max()-x.mean()), audio_out))
        if args.single:
            audio_out = np.concatenate(audio_out)
            audio_out = audio_out[None, :]

        saveAudioBatch(audio_out,
                       path=output_path,
                       basename=f'test_pitch_sweep_{i}', 
                       sr=config["transform_config"]["sample_rate"])
    print(f"Audio saved in {output_path}")
    print("FINISHED!\n")