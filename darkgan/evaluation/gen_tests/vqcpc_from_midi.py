import os

from datetime import datetime

from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioProcessor
from tqdm import trange
from mido import MidiFile
from data.loaders import get_data_loader
import random
import ipdb



def interpolate_batch(x, y, steps):
    alpha = 0
    output = []
    for i in np.linspace(0., 1., steps, True):
        output.append(x*(1 - i) + y*i)
    return torch.stack(output)

def spiral_traverse(z_dim, n_samples):
    pi = torch.Tensor([np.pi])
    z_batch = torch.randn(n_samples, z_dim)
    t1 = torch.linspace(-1, 1, n_samples)
    for i in range(n_samples):
        if i == 0:
            z_batch[0] = z_batch[0] / z_batch[0].norm()
            continue
        else:
            t2 = i/n_samples
            z_batch[i] = z_batch[0] + z_batch[0] * torch.sin(t2*2*pi) + z_batch[0] * torch.cos(t2*2*pi)
            z_batch[i] = z_batch[i]*t1[i]
    return z_batch

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


def radial_interpolation(input_z: int, n_samples: int=10):
    fixed_point = input_z
    fixed_point = fixed_point / fixed_point.norm()
    z_batch = torch.zeros(n_samples, input_z.size())

    t = torch.linspace(0.2, 8.0, n_samples)
    for k in range(n_samples):
        z_batch[k] = fixed_point * t[k]
    return z_batch

def generate(parser):
    parser.add_argument("--interpolate", dest="interp", action='store_true')
    parser.add_argument("--tick-duration", dest="tick", type=float, default=0.0006)
    args = parser.parse_args()
    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    transform_config = config['transform_config']
    loader_config = config['loader_config']

    processor = AudioProcessor(**transform_config)

    # transform_config['n_frames'] = 64
    processor2 = AudioProcessor(**transform_config)
    # processor2.audio_length *= 
    postprocess2 = processor2.get_postprocessor()

    if args.outdir == "":
        args.outdir = args.dir
    output_dir = mkdir_in_path(args.outdir, f"generation_samples")
    output_dir = mkdir_in_path(output_dir, f"from_midi")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))

    # loader_config["criteria"]["filter"]["seq_len"] = 64
    dbname = loader_config['dbname']
    loader = get_data_loader(dbname)(
        name=dbname + '_' + transform_config['transform'],
        preprocessing=processor, **loader_config)

    


    print("Loading MIDI file")
    
    midi_file = MidiFile(args.midi)
    midi_name = os.path.basename(args.midi).split('.')[0]
    pitch_list = []

    pitch_range = config['loader_config']['criteria']['filter']['pitch']
    pitch_cls_list = list(range(pitch_range[0], pitch_range[1]))
    track = list(midi_file.tracks)

    notes = list(filter(lambda x: x.type=="note_on" or x.type=="note_off", track[0]))
    notes_on = list(filter(lambda x: x.type=="note_on", track[0]))

    note_list = []
    offset = 0
    for i, note in enumerate(notes):

        noteoff = None
        offset += note.time
        
        if note.type == "note_off":
            continue
        if note.note > max(pitch_cls_list) or note.note < min(pitch_cls_list):
            continue
        duration = 0
        for j, n in enumerate(notes[i+1:]):
            if n.type == "note_off" and n.note == note.note:
                duration += n.time
                break
            duration += n.time
        note_list.append((note.note, duration, offset))

    tick = args.tick
    # tick = 0.0006  # s
    interp_step = 20
    total_duration = int((offset + duration) * tick * 16000)
    # total_duration = processor.audio_length*len(note_list)


    

    # z, _ = model.buildNoiseData(1, inputLabels=labels[0:1], skipAtts=True, test=True)
    
    # zs = radial_interpolation(128, interp_step)

    for y in range(3):
        orig_label = torch.Tensor(random.sample(loader.val_labels, k=len(note_list)))
        z_noise = torch.randn(128)
        output_buffer = torch.zeros(total_duration)
        k = 0
        spiral_z = spiral_traverse(model.config.noiseVectorDim, len(note_list))
        zx = torch.randn(128)
        zy = torch.randn(128)
        offset_samples = 0
        for i, note in enumerate(note_list):

            pitch, dur, offset = note

            # dur_frames = int(np.ceil(32*dur*tick))
            dur_frames = int(np.ceil(dur*tick*16000/512))
            # dur_frames = 32
            offset_samples = int(offset * tick * 16000)
            # offset_samples += 32*512

            ups = torch.nn.Upsample(size=(1, dur_frames), mode='nearest')
            cpcl = ups(orig_label[:, None, None, 2:])
            label = torch.cat([orig_label[:, :2], cpcl.reshape(-1, cpcl.size(-1))], dim=1)
            label[:, 1] = pitch_cls_list.index(pitch)

            z, _ = model.buildNoiseData(1, inputLabels=label[0:1], skipAtts=True, test=True)
            if args.interp:
                if k % (interp_step-1) == 0:
                    k=0
                    zs = interpolate_batch(zx, zy, interp_step)
                    zx = zy
                    zy = torch.randn(128)
                k+=1

                # zz = spiral_z[i][None, :, None, None]
                zz = zs[k, :, None, None]
                z[:, :128, :, :] = zz.repeat(1, 1, 1, z.size(-1))

            else:
                z[:, :128, :, :] = z_noise[None, :, None, None].repeat(1, 1, 1, z.size(-1))
                # z.transpose(1, 3)[:, :, :, :model.config.noiseVectorDim] = z_noise
            
            
            gnet = model.netG.eval()

            data_batch = []
            with torch.no_grad():

                synth_note = model.test(z, toCPU=True, getAvG=False).cpu()

                processor2.audio_length = dur_frames * 512
                # transform_config['n_frames'] = dur_frames
                # processor2.audio_length = 32 * 512
                # processor2 = AudioProcessor(**transform_config)
                postprocess2 = processor2.get_postprocessor()

                audio_out = list(map(postprocess2, synth_note))
                audio_out = list(map(processor2.pre_pipeline[2], audio_out))
                audio_out[0] = audio_out[0] / audio_out[0].max()
                # saveAudioBatch(audio_out,
                #                path=output_dir,
                #                basename=f'{midi_name}_{i}_interp_{args.interp}_test',
                #                sr=config["transform_config"]["sample_rate"])
                output_buffer[offset_samples:offset_samples + len(audio_out[0])] += audio_out[0]

        output_buffer = output_buffer / output_buffer.max()
        saveAudioBatch(output_buffer[None, :],
                       path=output_dir,
                       basename=f'{midi_name}_{y}_interp_{args.interp}_{datetime.today().strftime("%Y_%m_%d_%H")}',
                       sr=config["transform_config"]["sample_rate"])
    print(f"Audios saved in {output_dir}")
