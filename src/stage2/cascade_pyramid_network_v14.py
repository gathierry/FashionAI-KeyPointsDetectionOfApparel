import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, config, block, num_blocks, pretrained_model=None):
        super(GlobalNet, self).__init__()
        self.in_planes = 64
        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            # Bottom-up layers
            self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3

        # Lateral layers
        self.latlayer2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU())
        self.latlayer3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU())
        self.latlayer4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU())

        # Top-down layers
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer4 = nn.Conv2d(256, config.num_keypoints, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        # Top-down
        p4= self.latlayer2(c4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        p1 = self.toplayer4(p2)
        return p1, p2, p3, p4

class RefineNet(nn.Module):
    def __init__(self, config):
        super(RefineNet, self).__init__()
        self.bottleneck3 = nn.Sequential(Bottleneck(256, 64, 1),
                                         nn.ConvTranspose2d(256, 256, kernel_size=2 * 2, stride=2, padding=2 // 2))
        self.bottleneck4 = nn.Sequential(Bottleneck(256, 64, 1),
                                         Bottleneck(256, 64, 1),
                                         nn.ConvTranspose2d(256, 256, kernel_size=2 * 4, stride=4, padding=4 // 2))
        self.output = nn.Sequential(Bottleneck(256*3, 64, 1),
                                    nn.Conv2d(256, config.num_keypoints, kernel_size=1, stride=1, padding=0))

    def forward(self, p2, p3, p4):
        p3 = F.relu(self.bottleneck3(p3))
        p4 = F.relu(self.bottleneck4(p4))
        return self.output(torch.cat([p2, p3, p4], dim=1))

def GlobalNet50(config, pretrained=False):
    return GlobalNet(config, Bottleneck, [3, 4, 6, 3], torchvision.models.resnet50(pretrained=pretrained))

def GlobalNet101(config, pretrained=False):
    return GlobalNet(config, Bottleneck, [3, 4, 23, 3], torchvision.models.resnet101(pretrained=pretrained))

def GlobalNet152(config, pretrained=False):
    return GlobalNet(config, Bottleneck, [3, 8, 36, 3], torchvision.models.resnet152(pretrained=pretrained))

# remove c5
class CascadePyramidNetV14(nn.Module):
    def __init__(self, config):
        super(CascadePyramidNetV14, self).__init__()
        self.global_net = GlobalNet152(config, pretrained=True)
        self.refine_net = RefineNet(config)

    def forward(self, x):
        p1, p2, p3, p4 = self.global_net(x)
        out = self.refine_net(p2, p3, p4)
        return p1, out

if __name__ == '__main__':
    from torch.autograd import Variable
    from src.config import Config
    config = Config('outwear')
    net = CascadePyramidNetV14(config)
    fms = net(Variable(torch.randn(1, 3, 512, 512)))
