import torch
import numpy as np
from mmhyperspectral.utils import get_root_logger
from ..builder import PIPELINES


@PIPELINES.register_module()
class ExtractPatch:
    def __init__(self):
        self.val_ratio = 0.1

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
        gt_ = data_infos.get('gt', None)
        patch = data_infos.get('patch', None)
        train_indexes = data_infos.get('train_indexes', None)
        test_indexes = data_infos.get('test_indexes', None)
        total_indexes = data_infos.get('total_indexes', None)
        input_dimension = hsi.shape[-1]
        gt = gt_.reshape(np.prod(gt_.shape[:2]), )

        total_size = len(total_indexes)
        train_size = len(train_indexes)
        test_size = len(test_indexes)
        val_size = int(total_size * self.val_ratio)

        gt_hsi = gt[total_indexes] - 1
        gt_test_ = gt[test_indexes] - 1
        gt_train = gt[train_indexes] - 1
        gt_val = gt_test_[-val_size:]
        gt_test = gt_test_[:-val_size]

        hsi_ = self.select_patch(total_size, total_indexes, hsi, patch, pad_hsi, input_dimension)
        test_hsi_ = self.select_patch(test_size, test_indexes, hsi, patch, pad_hsi, input_dimension)
        train_hsi = self.select_patch(train_size, train_indexes, hsi, patch, pad_hsi, input_dimension)
        val_hsi = test_hsi_[-val_size:]
        test_hsi = test_hsi_[:-val_size]

        train_hsi = torch.from_numpy(train_hsi).type(torch.FloatTensor).unsqueeze(1)
        gt_train = torch.from_numpy(gt_train).type(torch.FloatTensor)

        val_hsi = torch.from_numpy(val_hsi).type(torch.FloatTensor).unsqueeze(1)
        gt_val = torch.from_numpy(gt_val).type(torch.FloatTensor)

        test_hsi = torch.from_numpy(test_hsi).type(torch.FloatTensor).unsqueeze(1)
        gt_test = torch.from_numpy(gt_test).type(torch.FloatTensor)

        hsi_ = torch.from_numpy(hsi_).type(torch.FloatTensor).unsqueeze(1)
        gt_hsi = torch.from_numpy(gt_hsi).type(torch.FloatTensor)

        return {'train_hsi': train_hsi, 'gt_train': gt_train}, {'val_hsi': val_hsi, 'gt_val': gt_val}, {
            'test_hsi': test_hsi, 'gt_test': gt_test, 'test_indexes': test_indexes}, \
               {'hsi': hsi_, 'gt_hsi': gt_hsi, 'total_indexes': total_indexes, 'gt': gt_}

    def __repr__(self):
        return self.__class__.__name__
