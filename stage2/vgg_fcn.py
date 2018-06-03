import torch
from torchvision.models import vgg
import torch.utils.model_zoo as model_zoo
from torch import nn

class VGG_FCN(nn.Module):
    def __init__(self, config, layer_num=16):
        super(VGG_FCN, self).__init__()
        self.config = config
        if layer_num == 16:
            vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
            scale = 16
        else:
            vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
            scale = 8
        vgg16_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        self.feature = vgg.make_layers(vgg16_cfg, batch_norm=False)
        feature_dict = self.feature.state_dict()
        pretrained_dict = model_zoo.load_url(vgg16_url)
        pretrained_dict = {k[9:]: v for idx, (k, v) in enumerate(pretrained_dict.items()) if idx < len(feature_dict.keys())}
        feature_dict.update_by_rule(pretrained_dict)

        self.feature.load_state_dict(feature_dict)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(512, self.config.num_keypoints, kernel_size=2*scale, stride=scale, padding=scale//2))

    def forward(self, x):
        x = self.feature(x)
        output = self.deconv(x)
        return output