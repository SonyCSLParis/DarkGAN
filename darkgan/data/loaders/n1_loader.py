from .base_loader import AudioDataLoader
from ..db_extractors.n1 import extract
import ipdb

class N1(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, criteria=self.criteria)
