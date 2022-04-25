# Copyright (c) OpenMMLab. All rights reserved.

from .train import init_random_seed, set_random_seed, train_model

__all__ = [
    'set_random_seed', 'init_random_seed', 'train_model'
]
