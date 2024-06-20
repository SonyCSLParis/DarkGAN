import os
import argparse
import sys
import torch

from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
from random import shuffle

from .base_db import get_base_db, get_hash_dict

import ipdb


__VERSION__ = "0.0.0"
criteria_keys = ['size', 'n_frames']
criteria_keys.sort()


def extract(data_path: str,
            dbname: str='n1', 
            criteria: dict={}):

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"
    if not os.path.exists(data_path):
        print(f'{data_path} not found')
        sys.exit(1)
    extraction_hash = get_hash_dict(criteria)
    data_path = data_path.rstrip('/')
    n_frames = criteria.get('n_frames')

    audio_path = os.path.join(data_path, 'audio')
    label_path = os.path.join(data_path, 'labels')

    description = get_base_db(dbname, __VERSION__)
    audio_files = list_files_abs_path(audio_path, '.mp3')
    audio_filenames = [get_filename(x) for x in audio_files]
    label_files = [os.path.join(label_path, x)  + '_wlbl' for x in audio_filenames]

    if len(audio_files) == 0:
        print('No wav files found!')
        sys.exit(1)
    size = criteria.get('size', len(audio_files))

    data = audio_files
    metadata = []
    label_count = {str(x): 0 for x in range(64)}
    for l in label_files:
        label_file = torch.load(l)['label_seq'].reshape(-1).tolist()[:n_frames]
        for v in label_file:
            label_count[str(v)] += 1
        metadata.append(label_file)

    description['attributes'] = {
        'label_seq': {
            'values': list(range(64)),
            'count': label_count,
            'loss': 'xentropy',
            'type': str(int)
        }
    }
    description['output_file'] = ''
    description['size'] = len(data)
    description['hash'] = extraction_hash
    return data, metadata, description


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nsynth database extractor')
    parser.add_argument('--wav', type=str, dest='path_wav',
                         help='Path to the nsynth root folder')

    parser.add_argument('--mp3', type=str, default='', dest='path_mp3',
                        help='Path to the nsynth root folder')

    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)

    
    args = parser.parse_args()
    if args.filter_config != None:
        fconfig = read_json(args.filter_config)
    else:
        fconfig = {}
    fconfig = {
        'bitrate': "16k",
        'size': 10
    }
    extract(path_wav=args.path_wav,
            path_mp3=args.path_mp3,
            criteria=fconfig)
