from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HyperSpectral(BaseDataset):
    def __init__(self):
        pass

    def load_annotations(self):
        mat_data = 