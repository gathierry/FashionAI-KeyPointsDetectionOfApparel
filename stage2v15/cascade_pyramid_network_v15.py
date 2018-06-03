import torch
import torch.nn as nn
import torch.nn.functional as F

from  stage2v15.nasnet import nasnetalarge

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalNet(nn.Module):
    def __init__(self, config):
        super(GlobalNet, self).__init__()
        pretrained_model = nasnetalarge(num_classes=1000, pretrained='imagenet')
        self.pm = pretrained_model

        # Lateral layers
        self.latlayer1 = nn.Conv2d(4032, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(2016, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(1008, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(168, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3, stride=1, padding=1)

    def features(self, input):
        x_conv0 = self.pm.conv0(input)
        x_stem_0 = self.pm.cell_stem_0(x_conv0)  # 168x128x128
        x_stem_1 = self.pm.cell_stem_1(x_conv0, x_stem_0)  # 336x64x64

        x_cell_0 = self.pm.cell_0(x_stem_1, x_stem_0)  # 1008x64x64
        x_cell_1 = self.pm.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.pm.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.pm.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.pm.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.pm.cell_5(x_cell_4, x_cell_3)  # 1008x64x64

        x_reduction_cell_0 = self.pm.reduction_cell_0(x_cell_5, x_cell_4)  # 1344x32x32

        x_cell_6 = self.pm.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.pm.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.pm.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.pm.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.pm.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.pm.cell_11(x_cell_10, x_cell_9)  # 2016x32x32

        x_reduction_cell_1 = self.pm.reduction_cell_1(x_cell_11, x_cell_10)  # 2688x16x16

        x_cell_12 = self.pm.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.pm.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.pm.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.pm.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.pm.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.pm.cell_17(x_cell_16, x_cell_15)  # 4032x16x16
        return x_stem_0, x_cell_5, x_cell_11, x_cell_17


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c2, c3, c4, c5 = self.features(x)

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        return p2, p3, p4, p5

class RefineNet(nn.Module):
    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck2 = Bottleneck(config.num_keypoints, 64, 1)
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1),
                                         nn.ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         nn.ConvTranspose2d(256, 256, kernel_size=2 * 4, stride=4, padding=4 // 2))
        self.bottleneck5 = nn.Sequential(Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         nn.ConvTranspose2d(256, 256, kernel_size=2 * 8, stride=8, padding=8 // 2))
        self.output = nn.Sequential(Bottleneck(1024, 64, 1),
                                    nn.Conv2d(256, config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4, p5):
        p2 = self.bottleneck2(p2)
        p3 = self.bottleneck3(p3)
        p4 = self.bottleneck4(p4)
        p5 = self.bottleneck5(p5)
        return self.output(torch.cat([p2, p3, p4, p5], dim=1))

class CascadePyramidNetV15(nn.Module):
    def __init__(self, config):
        super(CascadePyramidNetV15, self).__init__()
        self.global_net = GlobalNet(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out

if __name__ == '__main__':
    import torch
    from config import Config
    config = Config('outwear')
    net = CascadePyramidNetV15(config)
    fms = net(torch.randn(1,3,512, 512))
