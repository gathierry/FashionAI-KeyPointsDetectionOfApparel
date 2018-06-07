import torch
import torch.nn as nn
import torch.nn.functional as F

from  stage2v4.inceptionresnetv2 import inceptionresnetv2

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
        pretrained_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.conv2d_1a = pretrained_model.conv2d_1a  # downsampled
        self.conv2d_2a = pretrained_model.conv2d_2a
        self.conv2d_2b = pretrained_model.conv2d_2b
        self.maxpool_3a = pretrained_model.maxpool_3a  # downsampled
        self.conv2d_3b = pretrained_model.conv2d_3b
        self.conv2d_4a = pretrained_model.conv2d_4a
        self.maxpool_5a = pretrained_model.maxpool_5a  # downsampled
        self.mixed_5b = pretrained_model.mixed_5b
        self.repeat = pretrained_model.repeat
        self.mixed_6a = pretrained_model.mixed_6a  # downsampled
        self.repeat_1 = pretrained_model.repeat_1
        self.mixed_7a = pretrained_model.mixed_7a  # downsampled
        self.repeat_2 = pretrained_model.repeat_2
        self.block8 = pretrained_model.block8
        self.conv2d_7b = pretrained_model.conv2d_7b

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1088, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, config.num_keypoints, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        x = self.conv2d_1a(x)  # downsampled
        x = self.conv2d_2a(x)
        c1 = self.conv2d_2b(x)  # ch=64
        x = self.maxpool_3a(c1)  # downsampled
        x = self.conv2d_3b(x)
        c2 = self.conv2d_4a(x)  # ch=192
        x = self.maxpool_5a(c2)  # downsampled
        x = self.mixed_5b(x)
        c3 = self.repeat(x)  # ch=320
        x = self.mixed_6a(c3)  # downsampled
        c4 = self.repeat_1(x)  # ch=1088
        x = self.mixed_7a(c4)  # downsampled
        x = self.repeat_2(x)
        x = self.block8(x)
        c5 = self.conv2d_7b(x)  # ch=1536

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

class CascadePyramidNetV4(nn.Module):
    def __init__(self, config):
        super(CascadePyramidNetV4, self).__init__()
        self.global_net = GlobalNet(config)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p2, p3, p4, p5 = self.global_net(x)
        out = self.refine_net(p2, p3, p4, p5)
        return p2, out

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    from src.config import Config
    config = Config('outwear')
    net = CascadePyramidNetV4(config)
    fms = net(Variable(torch.randn(1,3,512, 512)))
