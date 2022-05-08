import logging
import numpy as np
from mmcv import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger('mmhyperspectralcls', log_file, log_level)



