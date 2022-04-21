import numpy as np
from sklearn import preprocessing
from mmhyperspectral.utils import get_root_logger
from ..builder import PIPELINES


@PIPELINES.register_module()
class Scale:
    def __init__(self):
        pass

    def __call__(self, data_infos):
        hsi = data_infos.get('hsi', None)
        gt = data_infos.get('gt', None)
        hsi = hsi.reshape(np.prod(hsi.shape[:2]), np.prod(hsi.shape[2:]))
        gt = gt.reshape(np.prod(gt.shape[:2]), )

        hsi = preprocessing.scale(hsi)
        hsi = hsi.reshape(hsi.shape[0], hsi.shape[1], hsi.shape[2])
        return {"hsi": hsi, "gt": gt}


@PIPELINES.register_module()
class Pad:
    def __init__(self, patch):
        self.patch = patch

    def __call__(self, data_infos):
        hsi = data_infos.get('hsi', None)
        gt = data_infos.get('gt', None)
        pad_hsi = np.lib.pad(hsi, ((self.patch, self.patch), (self.patch, self.patch),
                                   (0, 0)), 'constant', constant_values=0)
        return {'hsi': hsi, 'pad_hsi': pad_hsi, 'gt': gt, 'patch': self.patch}


@PIPELINES.register_module()
class Sampling:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, data_infos):
        self.logger = get_root_logger(log_level="INFO")
        gt = data_infos.get('gt', None)
        train = {}
        test = {}
        total = {}
        classes = max(gt)
        for i in range(classes):
            indexes = [j for j, x in enumerate(gt.ravel().tolist()) if x == i + 1]
            np.random.shuffle(indexes)
            total[i] = indexes
            if self.ratio != 1:
                nb_val = max(int((1 - self.ratio) * len(indexes)), 3)
            else:
                nb_val = 0
            train[i] = indexes[:-nb_val]
            test[i] = indexes[-nb_val:]
        train_indexes = []
        test_indexes = []
        total_indexes = []
        for i in range(classes):
            train_indexes += train[i]
            test_indexes += test[i]
            total_indexes += total[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        np.random.shuffle(total_indexes)

        self.logger.info(f"Train size:{len(train_indexes)},"
                         f"Val size:{int(total_indexes*self.ratio)},"
                         f"Test size:{total_indexes-train_indexes}")

        return {"hsi": data_infos['hsi'], 'pad_hsi': data_infos['pad_hsi'], 'gt': gt, 'patch': data_infos['patch'],
                'train_indexes': train_indexes, 'test_indexes': test_indexes, 'total_indexes': total_indexes}
