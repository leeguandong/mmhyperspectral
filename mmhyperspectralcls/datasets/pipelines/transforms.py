import numpy as np
from sklearn import preprocessing
from ..builder import PIPELINES


@PIPELINES.register_module()
class Scale:
    def __init__(self):
        pass

    def __call__(self, data_infos):
        hsi_info = data_infos['hsi_info']
        gt_label = data_infos['gt_label']
        hsi = hsi_info.reshape(np.prod(hsi_info.shape[:2]), np.prod(hsi_info.shape[2:]))
        gt = gt_label.reshape(np.prod(gt_label.shape[:2]), )

        hsi = preprocessing.scale(hsi)
        hsi = hsi.reshape(hsi_info.shape[0], hsi_info.shape[1], hsi_info.shape[2])
        return hsi, gt


@PIPELINES.register_module()
class Pad:
    def __init__(self, patch):
        self.patch = patch

    def __call__(self, data_infos):
        hsi, gt = data_infos
        hsi = np.lib.pad(hsi, ((self.patch, self.patch), (self.patch, self.patch), (0, 0)), 'constant',
                         constant_values=0)
        return hsi, gt


@PIPELINES.register_module()
class Sampling:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, data_infos):
        hsi, gt = data_infos
        train = {}
        test = {}
        labels_loc = {}
        classes = max(gt)
        for i in range(classes):
            indexes = [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]
            np.random.shuffle(indexes)
            labels_loc[i] = indexes
            if self.ratio != 1:
                nb_val = max(int((1 - self.ratio) * len(indexes)), 3)
            else:
                nb_val = 0
            train[i] = indexes[:-nb_val]
            test[i] = indexes[-nb_val:]
        train_indexes = []
        test_indexes = []
        for i in range(classes):
            train_indexes += train[i]
            test_indexes += test[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        return train_indexes, test_indexes
