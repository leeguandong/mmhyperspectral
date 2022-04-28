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

        hsi_ = hsi.reshape(np.prod(hsi.shape[:2]), np.prod(hsi.shape[2:]))
        # gt_ = gt.reshape(np.prod(gt.shape[:2]), )
        hsi_scale = preprocessing.scale(hsi_)

        hsi = hsi_scale.reshape(hsi.shape[0], hsi.shape[1], hsi.shape[2])
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
        data_infos.update({'pad_hsi': pad_hsi})
        data_infos.update({'patch': self.patch})
        return data_infos


@PIPELINES.register_module()
class Sampling:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, data_infos):
        self.logger = get_root_logger()
        gt_ = data_infos.get('gt', None)
        gt = gt_.reshape(np.prod(gt_.shape[:2]), )

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
                         f"Val size:{int(len(total_indexes)*self.ratio)},"
                         f"Test size:{len(total_indexes)-len(train_indexes)}")

        data_infos.update({'train_indexes': train_indexes})
        data_infos.update({'test_indexes': test_indexes})
        data_infos.update({'total_indexes': total_indexes})
        return data_infos
