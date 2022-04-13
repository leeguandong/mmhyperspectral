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