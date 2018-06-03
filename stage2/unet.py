import math

import torch
from torch import nn
import torch.nn.functional as F

from fpn import FPN101


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.fpn = FPN101(pretrained=True)
        self.reg_head = self._make_head(config.num_keypoints)
        # self.vis_head = self._make_head(3)
        self.freeze_bn()

    def forward(self, x):
        p3 = self.fpn(x)[0]
        reg = self.reg_head(p3)
        return reg

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU(True))
            # layers.append(nn.Dropout2d(p=0.5, inplace=True))
        final_conv = nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1)
        layers.append(final_conv)
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
