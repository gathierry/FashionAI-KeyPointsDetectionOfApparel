import math

import torch
import torch.nn as nn

from fpn import FPN101


class RetinaNet(nn.Module):
    def __init__(self, config, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN101(pretrained=True)
        if num_classes > 2:
            self.num_classes = num_classes + 1
        else:
            self.num_classes = 1
        self.reg_head = self._make_head(config.anchor_num * 4)
        self.cls_head = self._make_head(config.anchor_num * self.num_classes)
        # self.seg_head = self._make_head(config.num_keypoints)
        # self.vis_head = self._make_head(1)
        self.freeze_bn()

    def forward(self, x):
        fms = self.fpn(x)
        reg_preds = []
        cls_preds = []
        for fm in fms:
            reg_pred = self.reg_head(fm)
            cls_pred = self.cls_head(fm)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)  # [N,9*4,H,W] -> [N,H,W,9*4] -> [N,H*W*9,4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            reg_preds.append(reg_pred)
            cls_preds.append(cls_pred)
        return torch.cat(reg_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            conv.weight.data.normal_(0.0, 0.01)
            conv.bias.data.fill_(0)
            layers.append(conv)
            layers.append(nn.ReLU(True))
            # layers.append(nn.Dropout2d(p=0.5, inplace=True))
        final_conv = nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1)
        final_conv.weight.data.normal_(0.0, 0.01)
        final_conv.bias.data.fill_(-math.log((1-0.01)/0.01))
        layers.append(final_conv)
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == '__main__':
    from torch.autograd import Variable
    from time import time
    net = RetinaNet(80).cuda()
    t0 = time()
    loc_preds, cls_preds = net(Variable(torch.randn(1,3,600,900)).cuda())
    t1 = time()
    # print(loc_preds.size())
    # print(cls_preds.size())
    print(t1-t0)

    # loc_grads = Variable(torch.randn(loc_preds.size()))
    # cls_grads = Variable(torch.randn(cls_preds.size()))
    # loc_preds.backward(loc_grads)
    # cls_preds.backward(cls_grads)