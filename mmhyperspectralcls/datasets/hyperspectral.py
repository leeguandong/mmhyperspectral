import scipy.io as sio
from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module()
class HyperSpectral:
    def __init__(self,
                 manner,
                 data_prefix,
                 data_gt,
                 total_size,
                 split_ratio,
                 pipeline):
        self.manner = manner
        self.data_prefix = data_prefix
        self.data_gt = data_gt
        self.total_size = total_size
        self.split_ratio = split_ratio
        self.pipeline = Compose(pipeline)

        self.data_infos = self.load_annotations()
        self.train, self.val, self.test = self.prepare_data()
        self.train_dataset = self.train_dataset(self.train)
        self.val_dataset = self.val_dataset(self.val)
        self.test_dataset = self.test_dataset(self.test)

    def load_annotations(self):
        mat_data = sio.loadmat(self.data_prefix)
        mat_gt = sio.loadmat(self.data_gt)

        data_infos = {}
        if self.manner == 'IN':
            data_infos['hsi_info'] = mat_data['indian_pines_corrected']
            data_infos['gt_label'] = mat_gt['indian_pines_gt']
        elif self.manner == 'PaviaU':
            data_infos['hsi_info'] = mat_data['paviaU']
            data_infos['gt_label'] = mat_gt['paviaU_gt']
        elif self.manner == 'Pavia':
            data_infos['hsi_info'] = mat_data['pavia']
            data_infos['gt_label'] = mat_gt['pavia_gt']
        elif self.manner == 'Salinas':
            data_infos['hsi_info'] = mat_data['salinas_corrected']
            data_infos['gt_label'] = mat_gt['salinas_gt']
        elif self.manner == 'KSC':
            data_infos['hsi_info'] = mat_data['KSC']
            data_infos['gt_label'] = mat_gt['KSC_gt']
        elif self.manner == 'Botswana':
            data_infos['hsi_info'] = mat_data['Botswana']
            data_infos['gt_label'] = mat_gt['Botswana_gt']
        else:
            raise TypeError(f"{self.manner} is not support")
        return data_infos

    def prepare_data(self):
        return self.pipeline(self.data_infos)
