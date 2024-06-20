import math
import torch
import numpy as np
import torch.nn.functional as F

from utils.utils import *
from datetime import datetime
from evaluation.train_inception_model import SpectrogramInception3
from data.preprocessing import AudioProcessor

from ..inception_models import DEFAULT_INSTRUMENT_INCEPTION_MODEL, DEFAULT_INCEPTION_PREPROCESSING_CONFIG
from data.audio_transforms import MelScale
from tqdm import trange
from data.loaders import get_data_loader
from gans.ac_criterion import ACGANCriterion

import ipdb


class InceptionScore():
    def __init__(self):

        self.sumEntropy = 0
        self.sumSoftMax = None
        self.nItems = 0
        # self.classifier = classifier.eval()

    def updateWithMiniBatch(self, y):
        # y = self.classifier(ref).detach()
        if self.sumSoftMax is None:
            self.sumSoftMax = torch.zeros(y.size()[1]).to(y.device)

        # Entropy
        # x = F.softmax(y, dim=1, dtype=torch.float64) * F.log_softmax(y, dim=1)
        x = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        self.sumEntropy += x.sum().item()

        # Sum soft max
        self.sumSoftMax += F.softmax(y, dim=1).sum(dim=0)

        # N items
        self.nItems += y.size()[0]

    def getScore(self, eps=1e-06):
        x = self.sumSoftMax
        x = x * torch.log(x / self.nItems + eps)
        output = self.sumEntropy - (x.sum().item())
        output /= self.nItems
        return math.exp(output)


class InceptionScoreMaker(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), \
            f"Inception config {config_path} not found"

        self.config = read_json(config_path)
        self.model_path = self.config["path"]
        self.device = 'cuda' if GPU_is_available() else 'cpu'

        self.init_preprocessing()
        self.get_attribute_list()
        self.load_model_state_dict()

    def get_attribute_list(self):
        # dummy loader to get att_dict
        loader_config = self.config['loader_config']
        loader_config['criteria']['size'] = 1
        dbname = loader_config.pop('dbname')
        loader_module = get_data_loader(dbname)
        dummy_loader = loader_module(
            name=dbname + '_' + self.config['transform_config']['transform'], preprocessing=self.processor, **loader_config)
        att_dict = dummy_loader.header['attributes']
        
        self.att_classes = att_dict.keys()
        criterion = ACGANCriterion(att_dict)
        self.key_order = criterion.keyOrder
        self.att_size = criterion.attribSize

    def init_preprocessing(self):
        transform_config = self.config['transform_config']
        transform = transform_config['transform']
        print("Initializing IS pre-processing...")
        self.processor = AudioProcessor(**transform_config)
        # HACK: this should go to audio_transforms.py 
        self.mel = MelScale(sample_rate=transform_config['sample_rate'],
                       fft_size=transform_config['fft_size'],
                       n_mel=transform_config.get('n_mel', 256),
                       rm_dc=True)
        self.mel = self.mel.to(self.device)

    def load_model_state_dict(self):
        print(f"Loading inception model: {self.config['path']}")
        state_dict = torch.load(self.config['path'], map_location=self.device)
        self.inception_cls = SpectrogramInception3(state_dict['fc.weight'].shape[0], aux_logits=False)
        self.inception_cls.load_state_dict(state_dict)
        


        # ipdb.set_trace()
        # assert self.inception_cls.fc.out_features == sum(self.att_size), "Error: mismatch between inception output size and attributes total size"
        


        self.inception_cls = self.inception_cls.to(self.device)

    def __call__(self, path, attribute, batch_size=100, is_batch_size=5000):
        audio_files = list(list_files_abs_path(path, 'wav'))
        n_audio = len(audio_files)
        is_batch_size = min(n_audio, is_batch_size)
        start_idx, end_idx = self.get_attribute_idx(attribute)

        inception_score = []
        with torch.no_grad():
            pbar = trange(int(np.ceil(n_audio/is_batch_size)), desc="Inception Score outer-loop")
            for i in pbar:
                is_maker = InceptionScore()
                is_batch = audio_files[i*is_batch_size:is_batch_size*(i+1)]
                
                for j in range(int(np.ceil(is_batch_size/batch_size))):
                    batch = is_batch[j*batch_size:batch_size*(j+1)]
                    pred = self.pred_batch(batch)
                    pred = pred[:, start_idx:end_idx]

                    is_maker.updateWithMiniBatch(pred)
                    inception_score.append(is_maker.getScore())
                
                is_mean = np.mean(inception_score)
                is_std = np.std(inception_score)
                pbar.set_description("inception_score = {0:.4f} +- {1:.4f}".format(is_mean, is_std/2.))
        return {'mean_is': np.mean(inception_score), 'std_is': np.std(inception_score)}

    def get_attribute_idx(self, attribute):
        assert attribute in self.att_classes, f'Error: the selected attribute ({attribute}) is not in [{self.att_classes}]'

        att_idx = self.key_order.index(attribute)
        att_size = self.att_size[att_idx]
        start_idx = sum(self.att_size[:att_idx])

        return start_idx, start_idx + att_size

    def preprocess_batch(self, audio_batch):
        p_batch = self.processor(audio_batch)
        # take magnitude
        p_batch = torch.Tensor(p_batch)[:, 0:1].to(self.device)
        # mel transform
        p_batch = self.mel(p_batch)
        # interpolate
        p_batch  = F.interpolate(p_batch, (299, 299))
        return p_batch

    def pred_batch(self, audio_batch):
        p_batch = self.preprocess_batch(audio_batch)
        pred = self.inception_cls(p_batch.float()).detach().cpu()
        return pred


def test(parser):
    parser.add_argument('-c', dest="config", default="")
    parser.add_argument('--att', dest="attribute", required=True, type=str)
    # parser.add_argument('-b', '--batch-size', dest="batch_size", default=50)
    parser.add_argument('-N', dest='n_is', default=5000, 
        help="number of samples over which to compute IS")
    parser.add_argument('-n', '--name', dest='name', default='')
    args = parser.parse_args()

    inception_classifier = AudioInceptionClassifier(args.config)
    is_mean, is_std = inception_classifier(args.dir, attribute=args.att)
    
    output_path = os.path.join(args.dir, "evaluation_metrics")
    checkexists_mkdir(output_path)

    print("Computing inception score on true data...\nYou can skip this with ctrl+c")
    output_file = f'{output_path}/inception_score_{args.attribute}_{args.name}_{datetime.now().strftime("%d-%m-%y_%H_%M")}.txt'
    with open(output_file, 'w') as f:
        f.write(str(is_mean) + '\n')
        f.write(str(is_std))
        f.close()
