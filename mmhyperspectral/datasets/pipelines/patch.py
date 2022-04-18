import torch
import numpy as np
from mmhyperspectral.utils import get_root_logger
from ..builder import PIPELINES


@PIPELINES.register_module()
class ExtractPatch:
    def __init__(self):
        pass

    def index_assignment(self, hsi_indexes, col, patch):
        new_assign = {}
        for counter, value in enumerate(hsi_indexes):
            assign_0 = value // col + patch
            assign_1 = value % col + patch
            new_assign[counter] = [assign_0, assign_1]
        return new_assign

    def select_patch(self, hsi_size, hsi_indexes, hsi, patch, pad_hsi, dimension):
        patch_data = np.zeros((hsi_size, 2 * patch + 1, 2 * patch + 1, dimension))
        hsi_assign = self.index_assignment(hsi_indexes, hsi.shape[1], patch)
        for i in range(len(hsi_assign)):
            selected_rows = pad_hsi[range(hsi_assign[i][0] - patch, hsi_assign[i][0] + patch + 1)]
            patch_data[i] = selected_rows[:, range(hsi_assign[i][1] - patch, hsi_assign[i][1] + patch + 1)]
        return patch_data

    def __call__(self, data_infos):
        hsi = data_infos.get('hsi', None)
        pad_hsi = data_infos.get('pad_hsi', None)
        gt = data_infos.get('gt', None)
        patch = data_infos.get('patch', None)
        train_indexes = data_infos.get('train_indexes', None)
        test_indexes = data_infos.get('test_indexes', None)
        total_indexes = data_infos.get('total_indexes', None)
        input_dimension = hsi.shape[-1]
        total_size = len(total_indexes)
        train_size = len(train_indexes)
        test_size = len(test_indexes)
        val_size = train_size - test_size

        gt_ = gt[total_indexes] - 1
        gt_train = gt[train_indexes] - 1
        gt_test = gt[test_indexes] - 1

        hsi_ = self.select_patch(total_size, total_indexes, hsi, patch, pad_hsi, input_dimension)
        train_hsi = self.select_patch(train_size, train_indexes, hsi, patch, pad_hsi, input_dimension)
        test_hsi_ = self.select_patch(test_size, test_indexes, hsi, patch, pad_hsi, input_dimension)

        val_hsi = test_hsi_[-val_size:]
        gt_val = gt_test[-val_size:]

        test_hsi = test_hsi_[:-val_size]
        gt_test = gt_test[:-val_size]

        train_hsi = torch.from_numpy(train_hsi).type(torch.FloatTensor).unsqueeze(1)
        gt_train = torch.FloatTensor()