from mmcv.runner import BaseModule

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class Residual(BaseModule):
    def __init__(self,
                 in_channels,
                 ):
        pass


@BACKBONES.register_module()
class SSRN(BaseBackbone):
    def __init__(self,
                 band,
                 base_channels=24,
                 num_stages=4,
                 out_indices=(3,),
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv3d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm3d', 'GroupNorm3d'])
                 ],
                 drop_path_rate=0.0):
        super(SSRN, self).__init__(init_cfg)
        self.band = band
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual


    def _freeze_stages(self):
        if self.frozen_stages>=0:
            self.





    def forward(self, x):
        pass

    def train(self, mode=True):
        super(SSRN, self).train(mode)
        self._f
