import os
import argparse
import hashlib
import sys
import pickle
import requests

from .nsynth_tools import audioset_skip, audioset_cnn14_ap
from utils.utils import get_date, mkdir_in_path, read_json, list_files_abs_path, get_filename, save_json
from random import shuffle
from tqdm import trange, tqdm

from .base_db import get_base_db
from numpy import argsort
import ipdb
from librosa.core import load
import torch

from panns_inference import AudioTagging, labels

__VERSION__ = "0.0.0"
MAX_N_FILES_IN_FOLDER = 10000

nsynth_keys = ['audioset_pred', 'instrument_family_str', 'instrument', 'instrument_source_str', \
               'pitch', 'qualities_str', 'velocity', 'vqcp_codes']

from_nsynth_keys = {
    'instrument_family_str': 'instrument',
    'instrument': 'instrument_id',
    'instrument_source_str': 'instrument_type',
    'pitch': 'pitch',
    'qualities_str': 'properties',
    'velocity': 'velocity',
    'audioset_pred': 'audioset',
    'vqcp_codes': 'vqcp_codes'
}

nsynth_train_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz'
nsynth_valid_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz'
nsynth_test_url = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz'


def get_hash_dict(_dict: dict):
    keys = list(_dict.keys())
    keys.sort()
    hash_list = []
    for k in keys:
        # TODO: list here will break if the list is of objects
        if type(_dict[k]) in [list, str, int, float, bool]:
            hash_list.append((k, _dict[k]))
        elif type(_dict[k]) is dict:
            hash_list.append(get_hash_dict(_dict[k]))
    return hashlib.sha1(str(hash_list).encode()).hexdigest()


def get_standard_format(path: str, dbname='nsynth', codebook_size=None, cpc_path=None):
    nsynth_description = get_base_db('nsynth', __VERSION__)
    description_file = os.path.join(path, f'{dbname}.json')
    if os.path.exists(description_file):
        print(f"Description file {description_file} exists. Reloading.")
        return read_json(description_file)

    nsynth_metadata = os.path.join(path, 'examples_audioSet.json')
    nsynth_audio    = os.path.join(path, 'audio')

    # CHANGE IN CASE CPC ENCODINGS CHANGE
    # nsynth_cpc      = os.path.join(path, 'CPC/v2/base_config_carotte_2020-10-26_19:04:35')
    # nsynth_cpc      = os.path.join(path, 'CPC/v2/base_config_carotte_128_cl_64_dim')
    assert os.path.exists(cpc_path), f"Path {cpc_path} doesn't exist!"
    nsynth_cpc      = cpc_path

    root_dir = mkdir_in_path(path, f'nsynth_standard')

    extraction_config = os.path.join(root_dir, 'extraction_config.json')
    if os.path.exists(extraction_config):
        print("Extraction configuration exists. Loading...")
        return read_json(extraction_config)
    print(f"Creating new extraction NSYNTH description config ({description_file}).")

    metadata = read_json(nsynth_metadata)
    nsynth_files = list_files_abs_path(nsynth_audio, '.wav')

    nsynth_description['total_size'] = len(nsynth_files)
    attributes = {}
    n_folders = 0
    nsynth_files = list(filter(lambda x: get_filename(x) in metadata, nsynth_files))
    pbar = tqdm(enumerate(nsynth_files), desc='Reading files')
    for i, file in pbar:
        if i % MAX_N_FILES_IN_FOLDER == 0:
            n_folders += 1
            output_dir = mkdir_in_path(root_dir, f'folder_{n_folders}')

        filename = get_filename(file)
        output_file = os.path.join(output_dir, filename + '.json')
        nsynth_description['data'].append(output_file)

        item = metadata[filename]
        out_item = {
            'path': file,
            'attributes': {}
        }
        for att in nsynth_keys:
            myatt = from_nsynth_keys[att]
            val = None
            if att in item:
                val = item[att]
            if myatt not in attributes:
                if type(val) in [str, int, list]:
                    attributes[myatt] = {
                        'values': [],
                        'count': {},
                        'type': str(type(item[att]))
                    }
                elif myatt == 'audioset':
                    keys = list(val.keys())
                    keys.sort()
                    attributes['audioset'] = {
                        k: {'type': str(float),
                            'max': 0.0,
                            'min': 1.0,
                            'mean': 0.0,
                            'var': 0.0} for k in keys}

                elif myatt == 'vqcp_codes':
                    # hardcoded
                    attributes['vqcp_codes'] = {
                        'values': list(range(codebook_size)),
                        'count': {str(x): 0 for x in range(codebook_size)},
                        'loss': 'xentropy',
                        'type': (str(list), str(int)),
                        'seq_len': 32
                        # 'type': str(list)
                    }
            if type(val) in [int, str]:
                if val not in attributes[myatt]['values']:
                    attributes[myatt]['values'].append(val)
                    attributes[myatt]['count'][str(val)] = 0
                attributes[myatt]['count'][str(val)] += 1
            if att == 'qualities_str':
                for q in val:
                    if q == "reverb":
                        continue
                    if q not in attributes[myatt]['values']:
                        attributes[myatt]['values'].append(q)
                        attributes[myatt]['count'][str(q)] = 0
                    attributes[myatt]['count'][str(q)] += 1
            if myatt == 'audioset':
                for k in attributes[myatt]:
                    if val[k] > attributes[myatt][k]['max']:
                        attributes[myatt][k]['max'] = val[k]            
                    if val[k] < attributes[myatt][k]['min']:
                        attributes[myatt][k]['min'] = val[k]
                    attributes[myatt][k]['mean'] += val[k]

            if myatt == 'vqcp_codes':
                file_cpc = os.path.join(nsynth_cpc, filename + '_wlbl')
                if os.path.exists(file_cpc):  
                    label_seq = torch.load(file_cpc, map_location=torch.device('cpu'))['label_seq']
                    
                    val = label_seq.reshape(-1).tolist()
                    for v in val:
                        attributes['vqcp_codes']['count'][str(v)] += 1
                else:
                    print(f"VQ-CP codes not found. File {file_cpc} does not exist!")
                    val = str(None)

            out_item['attributes'][myatt] = val
        save_json(out_item, output_file)

    audioset_order = []
    as_keys = list(attributes['audioset'].keys())
    for i, k in enumerate(as_keys):
        attributes['audioset'][k]['mean'] /= nsynth_description['total_size']

        attributes['audioset'][k]['gmean'] = \
            (attributes['audioset'][k]['mean']*audioset_cnn14_ap[k])**(0.5)

    nsynth_description['attributes'] = attributes

    save_json(nsynth_description, description_file)
    return nsynth_description


def extract(path: str, criteria: dict={}, download: bool=False):
    criteria_keys = ['filter', 'balance', 'attributes', 'size', 'audioset', 'codebook_size', 'cpc_path']

    if criteria != {}:
        assert all(k in criteria_keys for k in criteria),\
            "Filter criteria not understood"
    if download:
        # downloading
        nsynth_dir = get_filename(path)
        nsynth_tar = requests.get(nsynth_train_url)
        with open(os.path.join(path, 'nsynth.tar.gz'), 'wb') as file:
            file.write(nsynth_tar.content)
            file.close()
    elif not os.path.exists(path):
        print('NSynth folder not found')
        sys.exit(1)

    root_dir = mkdir_in_path(path, f'extractions')
    extraction_hash = get_hash_dict(criteria)

    extraction_dir = mkdir_in_path(root_dir, str(extraction_hash))
    data_file = os.path.join(extraction_dir, 'data.pt')
    desc_file = os.path.join(extraction_dir, 'extraction.json')
    if os.path.exists(data_file):
        print(f"Extraction file {data_file} exists. Reloading.")
        extraction_desc = read_json(desc_file)

        print(f"Loading {extraction_desc['name']}\n" \
              f"Version: {extraction_desc['version']}\n" \
              f"Date: {extraction_desc['date']}\n")
        return pickle.load(open(data_file, 'rb'))
    print(f"New extraction criteria:\n{criteria}\nCreating file {data_file}.")
    nsynth_standard_desc = get_standard_format(
        path, 
        codebook_size=criteria.get('codebook_size', 0), 
        cpc_path=criteria.get('cpc_path', ''))

    extraction_dict = get_base_db('nsynth', __VERSION__)
    attribute_list = list(nsynth_standard_desc['attributes'].keys())
    out_attributes = criteria.get('attributes', attribute_list)
    out_attributes.sort()

    # ordering of audioset attributes
    if 'audioset' in out_attributes:
        audioset_order = []
        audioset_keys = list(nsynth_standard_desc['attributes']['audioset'].keys())
        for _, as_att in nsynth_standard_desc['attributes']['audioset'].items():

            if criteria['filter'].get('as_gm', True):
                audioset_order.append(as_att['gmean'])
            else:
                audioset_order.append(as_att['mean'])
        audioset_order = argsort(audioset_order)[::-1]
        audioset_order = [audioset_keys[i] for i in audioset_order]

    # get database attribute values and counts 
    # given the filtering criteria
    attribute_dict = {att: {'values': [], 'count': {}, 'type': ''} for att in out_attributes} 
    for att in attribute_dict.keys():
        if att in criteria.get('filter', {}).keys(): 
            if att in ['pitch', 'instrument_id']:

                attribute_dict[att]['values'] = list(range(*criteria['filter'][att]))
            elif att != 'audioset':
                criteria['filter'][att].sort()
                attribute_dict[att]['values'] = criteria['filter'][att]
        elif att != 'audioset':
            attribute_dict[att] = nsynth_standard_desc['attributes'][att].copy()

        if att == 'audioset':
            filtered_as_keys = [att for att in audioset_order if att not in audioset_skip]
            n_as_atts = len(filtered_as_keys)
            
            # if 'filter' in criteria:
            #     n_as_atts = criteria['filter'].get('audioset', n_as_atts)
            # as_out_keys = filtered_as_keys[:n_as_atts]

            as_out_keys = filtered_as_keys
            as_out_keys.sort()
            attribute_dict[att]['loss'] = 'bce'
            attribute_dict[att]['type'] = str(float)
            attribute_dict[att]['values'] = as_out_keys
            attribute_dict[att]['var'] = {k: 0.0 for k in as_out_keys}
            attribute_dict[att]['mean'] = {k: nsynth_standard_desc['attributes'][att][k]['mean'] for k in as_out_keys}
            attribute_dict[att]['gmean'] = {k: nsynth_standard_desc['attributes'][att][k]['gmean'] for k in as_out_keys}
            attribute_dict[att]['max'] = {k: nsynth_standard_desc['attributes'][att][k]['max'] for k in as_out_keys}
            attribute_dict[att]['min'] = {k: nsynth_standard_desc['attributes'][att][k]['min'] for k in as_out_keys}
            
        else:
            attribute_dict[att]['loss'] = 'xentropy'
            if att == 'properties':
                attribute_dict[att]['loss'] = 'bce'
            attribute_dict[att]['type'] = nsynth_standard_desc['attributes'][att]['type']
            attribute_dict[att]['values'].sort()
            attribute_dict[att]['count'] = {str(k): 0 for k in attribute_dict[att]['values']}

    size = criteria.get('size', nsynth_standard_desc['total_size'])
    balance = False
    
    if 'balance' in criteria:
        balance = True
        b_atts = criteria['balance']

        for b_att in b_atts:
            count = []
            if b_att in attribute_dict:
                b_att_vals = attribute_dict[b_att]['values']
            else:
                if b_att in criteria.get('filter', {}):
                    b_att_vals = criteria['filter'][b_att]
                else:
                    b_att_vals = nsynth_standard_desc['attributes'][b_att]['values']
            for v in b_att_vals:
                
                count.append(nsynth_standard_desc['attributes'][b_att]['count'][str(v)])
            n_vals = len(count)
            size = min(size, n_vals * min(count))

    data = []
    metadata = []
    pbar = tqdm(nsynth_standard_desc['data'])

    # hack audioset
    if 'audioset' in criteria:
        model = criteria['audioset'].get('model', 'Cnn14_16k_mAP=0.438.pth')
        temp = criteria['audioset'].get('temperature', 1)

        label_idx = [labels.index(l) for l in as_out_keys]
        sample_rate = 16000 if model == 'Cnn14_16k_mAP=0.438.pth' else 32000
        at = AudioTagging(checkpoint_path=f"/home/admin/developer/panns_inference/panns_data/{model}", device='cuda')



    for file in pbar:
        item = read_json(file)

        item_atts = item['attributes']
        item_path = item['path']

        # skip files that do not comply with
        # filtered attribute criteria
        skip = False
        for att, val in item_atts.items():
        # for att in out_attributes:
            # val = item_atts[att]
            if att == 'audioset':
                continue
            elif att == 'vqcp_codes' and att in out_attributes:
                if val == str(None):
                    skip = True
                    print(f"Skipping file {file}, does not have CPC annotations")
                    break

            if att in criteria.get('filter', {}):
                if att not in ['pitch', 'instrument_id', 'audioset']:
                    if val not in criteria['filter'][att]:
                        skip = True
                        break
            if att not in attribute_dict:
                continue

            if type(val) is list:
                if any(v in attribute_dict[att]['values'] for v in val) or val == []:
                    continue

            if val not in attribute_dict[att]['values']: 
                skip = True
                break


        if skip: continue

        # check balance of attributes
        if balance:
            for b_att in b_atts:
                val = item_atts[b_att]
                if b_att in attribute_dict:
                    bsize = size / len(attribute_dict[b_att]['values'])
                elif b_att in criteria.get('filter', {}):
                    bsize = size / len(criteria['filter'][b_att])
                
                else:
                    bsize = size / len(nsynth_standard_desc['attributes'][b_att]['values'])

                if attribute_dict[b_att]['count'][str(val)] >= bsize:
                    skip = True
            if skip:
                continue
        # store attribute index in list
        data_item = []
        for att in out_attributes:
            val = item_atts[att]
            # if attribute is multi-label (n out of m)
            if att == 'vqcp_codes':
                n_frames = criteria['filter']['seq_len']
                file_cpc = os.path.join(criteria['cpc_path'], get_filename(item_path) + '_wlbl')
                if os.path.exists(file_cpc):  
                    label_seq = torch.load(file_cpc, map_location=torch.device('cpu'))['label_seq']
                    val = label_seq.reshape(-1).tolist()
                    data_val = val[:n_frames]
                else:
                    print(f"VQ-CP codes not found. File {file_cpc} does not exist!")
                    val = str(None)

            if type(val) is list:
                if all(isinstance(n, str) for n in val):
                    # we now consider binary attributes (1 or 0)
                    data_val = [0] * len(attribute_dict[att]['values'])
                    for v in val:
                        if v in attribute_dict[att]['values']:
                            idx = attribute_dict[att]['values'].index(v)
                            attribute_dict[att]['count'][str(v)] += 1
                            data_val[idx] = 1
                        else:
                            continue
                # TODO: consider float values (audioset) 
                # --> counts and value tracking makes no sense
                elif all(isinstance(n, float) for n in val):
                    pass
            
            elif att == 'audioset':
                audio, sr = load(item_path, sr=sample_rate)
                pred, _ = at.inference(audio[None, :], T=temp)
                pred = pred[0][label_idx]
                data_val = []
                for i, v in enumerate(attribute_dict[att]['values']):
                    as_val = item_atts[att][v]
                    as_val = pred[i]
                    data_val.append(as_val)
                    attribute_dict[att]['var'][v] += \
                        (nsynth_standard_desc['attributes'][att][v]['mean'] - as_val)**2


            else:
                idx = attribute_dict[att]['values'].index(val)
                attribute_dict[att]['count'][str(val)] += 1
                data_val = [idx]
                # data_val = idx
            data_item += data_val
            # data_item.append(data_val)
        if skip: continue

        data.append(item_path)
        metadata.append(data_item)
        extraction_dict['data'].append(file)
        if len(data) >= size:
            pbar.close()
            break

    if 'audioset' in attribute_dict:
        if criteria['filter'].get('quantile_gm', False):
            import numpy as np
            n_as_atts = criteria['filter'].get('audioset', n_as_atts)
            metadata = np.array(metadata)
            feats = metadata[:, :-1]
            q90 = np.quantile(feats, 0.8, axis=0)

            gmean_q_prec = np.array([(audioset_cnn14_ap[k]*q90[i])**0.5 for i, k in enumerate(as_out_keys)])
            q90args = gmean_q_prec.argsort()[::-1]
            fk = np.array(as_out_keys)[q90args][:n_as_atts]
            fk.sort()
            args_out = [as_out_keys.index(k) for k in fk]
            
            attribute_dict['audioset']['q80'] = list(q90[args_out])

            metadata = list(np.concatenate([feats[:, args_out], metadata[:, -1:]], axis=1))
            metadata = [list(x) for x in metadata]

            attribute_dict['audioset']['values'] = list(fk)

        for k in attribute_dict['audioset']['values']:
            attribute_dict['audioset']['var'][k] /= (size - 1) # variance with bias correction (delta degrees of freedom, ddof = 1)

    extraction_dict['attributes'] = attribute_dict
    extraction_dict['output_file'] = data_file
    extraction_dict['size'] = len(data)
    extraction_dict['hash'] = extraction_hash

    with open(data_file, 'wb') as fp:
        pickle.dump((data, metadata, extraction_dict), fp)
    save_json(extraction_dict, desc_file)
    return data, metadata, extraction_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nsynth database extractor')
    parser.add_argument('nsynth_path', type=str,
                         help='Path to the nsynth root folder')
    
    parser.add_argument('-f', '--filter', help="Path to extraction configuration",
                        type=str, dest="filter_config", default=None)
    
    parser.add_argument('--download', action='store_true', 
                        help="Download nsynth?",
                        dest="download", default=False)
    
    args = parser.parse_args()
    if args.filter_config != None:
        fconfig = read_json(args.filter_config)
    else:
        fconfig = {}
    # fconfig = {
    #     'attributes': ['pitch'],
    #     'balance': ['pitch'],
    #     'filter': {
    #         'instrument': ['mallet'],
    #         'pitch': [30, 50]
    #     }
    # }
    extract(path=args.nsynth_path,
            criteria=fconfig,
            download=args.download)
