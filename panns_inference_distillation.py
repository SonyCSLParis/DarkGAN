import torch


from panns_inference import AudioTagging
from panns_inference.panns_inference.pytorch_utils import move_data_to_device
from panns_inference.panns_inference.models import Cnn14


class Cnn14_T(Cnn14):

    def forward(self, input, mixup_lambda=None, T=1):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x)/T)
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class AudioTaggingTemp(AudioTagging):
	def __init__(self, **kwargs):

        self.model = Cnn14_T(sample_rate=32000, window_size=1024, 
            hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
            classes_num=self.classes_num)
        super(AudioTaggingTemp, self).__init__(**kwargs)

    def inference(self, audio, T=1):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None, T=T)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()

        return clipwise_output, embedding