import scipy.io as sio
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .pipelines import Compose
from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class HyperSpectral:
    def __init__(self,
                 manner: str = 'IN',
                 data_prefix: str = None,
                 data_gt: str = None,
                 pipeline: Sequence = ()):
        self.manner = manner
        self.data_prefix = data_prefix
        self.data_gt = data_gt
        self.pipeline = Compose(pipeline)

        self.data_infos = self.load_annotations()
        self.train, self.val, self.test, self.total = self.prepare_data()
        self.train_dataset = self.train_dataset(self.train)
        self.val_dataset = self.val_dataset(self.val)
        self.test_dataset = self.test_dataset(self.test)
        self.dataset = self.dataset(self.total)

    def load_annotations(self):
        mat_data = sio.loadmat(self.data_prefix)
        mat_gt = sio.loadmat(self.data_gt)

        data_infos = {}
        if self.manner == 'IN':
            data_infos['hsi'] = mat_data['indian_pines_corrected']
            data_infos['gt'] = mat_gt['indian_pines_gt']
        elif self.manner == 'PaviaU':
            data_infos['hsi'] = mat_data['paviaU']
            data_infos['gt'] = mat_gt['paviaU_gt']
        elif self.manner == 'Pavia':
            data_infos['hsi'] = mat_data['pavia']
            data_infos['gt'] = mat_gt['pavia_gt']
        elif self.manner == 'Salinas':
            data_infos['hsi'] = mat_data['salinas_corrected']
            data_infos['gt'] = mat_gt['salinas_gt']
        elif self.manner == 'KSC':
            data_infos['hsi'] = mat_data['KSC']
            data_infos['gt'] = mat_gt['KSC_gt']
        elif self.manner == 'Botswana':
            data_infos['hsi'] = mat_data['Botswana']
            data_infos['gt'] = mat_gt['Botswana_gt']
        else:
            raise TypeError(f"{self.manner} is not support")
        return data_infos

    def prepare_data(self):
        return self.pipeline(self.data_infos)

    def train_dataset(self, train_data):
        train_hsi = train_data.get('train_hsi', None)
        gt_train = train_data.get('gt_train', None)
        train_dataset = BaseDataset(train_hsi, gt_train)
        return train_dataset

    def val_dataset(self, val_data):
        val_hsi = val_data.get('val_hsi', None)
        gt_val = val_data.get('gt_val', None)
        val_dataset = BaseDataset(val_hsi, gt_val)
        return val_dataset

    def test_dataset(self, test_data):
        test_hsi = test_data.get('test_hsi', None)
        gt_test = test_data.get('gt_test', None)
        test_dataset = BaseDataset(test_hsi, gt_test)
        return test_dataset

    def dataset(self, hsi_data):
        hsi = hsi_data.get('hsi', None)
        gt_hsi = hsi_data.get('gt_hsi', None)
        total_indexes = hsi_data.get('total_indexes', None)
        dataset = BaseDataset(hsi, gt_hsi)
        dataset.total_indexes = total_indexes
        return dataset
