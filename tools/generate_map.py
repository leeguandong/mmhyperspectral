import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import numpy as np
from pathlib import Path

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmhyperspectral.apis import set_random_seed, train_model, test_model
from mmhyperspectral.datasets import build_dataset, build_dataloader
from mmhyperspectral.models import build_classifier
from mmhyperspectral.core import wrap_fp16_model
from mmhyperspectral.utils import list_to_colormap, classification_map, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='mmhyperspectral generate map')
    parser.add_argument('--config', default='../results/resnet50_in.py', help='test config file path')
    parser.add_argument('--checkpoint', default='../results/latest.pth', help='checkpoint file')
    parser.add_argument('--out', default='../results', help='output result file')
    parser.add_argument(
        '--device', choices=['cpu', 'cuda'], default='cpu', help='device used for testing')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(args.out))

    base_dataset = build_dataset(cfg.data.train)
    total_dataset = base_dataset.dataset
    total_indexes = total_dataset.total_indexes
    gt_hsi = total_dataset.gt
    gt = gt_hsi.flatten()
    hsi_label = np.zeros(gt.shape)

    data_loader = build_dataloader(
        total_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        round_up=True)

    model = build_classifier(cfg.model)
    model.eval()

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location=args.device)

    results = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            result = model(return_loss=False, **data)
            results.extend(result)
    pred_label = np.argmax(np.vstack(results), axis=1)

    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            hsi_label[i] = 16
    gt = gt[:] - 1
    hsi_label[total_indexes] = pred_label
    pred = np.ravel(hsi_label)

    pred_list = list_to_colormap(pred)
    gt = list_to_colormap(gt)
    pred_list = np.reshape(pred_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt = np.reshape(gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(pred_list, gt_hsi, 300,
                       args.out + '/' + Path(args.config).stem + '.png')
    classification_map(gt, gt_hsi, 300,
                       args.out + '/' + Path(args.config).stem + '_gt.png')


if __name__ == "__main__":
    main()
