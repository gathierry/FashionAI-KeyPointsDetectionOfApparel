import torch
from torch import nn
import torch.nn.functional as F
import math

class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class ResidualUNet2D(nn.Module):
    def __init__(self, config):
        super(ResidualUNet2D, self).__init__()
        self.featureNum_forw = [3, 32, 64, 128, 256, 512]
        num_blocks_forw = [2, 2, 3, 3, 3]
        for i in range(1, len(self.featureNum_forw)):
            blocks = []
            for j in range(num_blocks_forw[i - 1]):
                if j == 0:
                    blocks.append(PostRes2d(self.featureNum_forw[i - 1], self.featureNum_forw[i]))
                else:
                    blocks.append(PostRes2d(self.featureNum_forw[i], self.featureNum_forw[i]))
            setattr(self, 'forw' + str(i), nn.Sequential(*blocks))
        for i in range(len(self.featureNum_forw) - 2, 0, -1):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes2d(self.featureNum_forw[i + 1] + self.featureNum_forw[i], self.featureNum_forw[i]))
                else:
                    blocks.append(PostRes2d(self.featureNum_forw[i], self.featureNum_forw[i]))
            setattr(self, 'back' + str(i), nn.Sequential(*blocks))
            upsample = [
                nn.ConvTranspose2d(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm3d(self.featureNum_forw[i + 1]),
                nn.ReLU(inplace=True)
            ]
            setattr(self, 'upsample' + str(i), nn.Sequential(*upsample))
        self.output = nn.Sequential(nn.Conv2d(32, config.num_keypoints, kernel_size=1))
        for i in range(1, 4):
            setattr(self, 'bridge_drop' + str(i), nn.Dropout2d(p=0.5, inplace=False))

    def forward(self, x):
        conv1 = self.forw1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.forw2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)
        conv3 = self.forw3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)
        conv4 = self.forw4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2, stride=2)
        conv5 = self.forw5(pool4)
        up5 = self.upsample4(conv5)
        cat5 = torch.cat((up5, self.bridge_drop1(conv4)), 1)
        conv6 = self.back4(cat5)
        up6 = self.upsample3(conv6)
        cat6 = torch.cat((up6, self.bridge_drop2(conv3)), 1)
        conv7 = self.back3(cat6)
        up7 = self.upsample2(conv7)
        cat7 = torch.cat((up7, self.bridge_drop3(conv2)), 1)
        conv8 = self.back2(cat7)
        up8 = self.upsample1(conv8)
        cat8 = torch.cat((up8, self.bridge_drop3(conv1)), 1)
        conv9 = self.back1(cat8)
        out = self.output(conv9)
        return out

if __name__ == '__main__':
    from torch.autograd import Variable
    # net = ResNet(Bottleneck, [3, 4, 6, 3])
    # net.forward(Variable(torch.zeros([1, 3, 224, 224])))
    net = ResidualUNet2D()
    out = net.forward(Variable(torch.zeros([8, 1, 512, 512])))
    # print(out.size())