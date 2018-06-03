import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import torch.nn.functional as F

import pytorch_utils
from config import Config
from kpda_parser import KPDA
from stage2.unet import UNet
from stage2.vgg_fcn import VGG_FCN
from stage2.residual_unet import ResidualUNet2D
from stage2.cascade_pyramid_network import CascadePyramidNet
from stage2.keypoint_encoder import KeypointEncoder
from utils import draw_heatmap, draw_keypoints, normalized_error
from stage2.data_generator import DataGenerator

root_path = '/home/storage/lsy/fashion/'
db_path = root_path + 'FashionAI_Keypoint_Detection/train/'

def print_heatmap(hm, img, config):
    scale = 255 // config.hm_alpha
    hm = (hm * scale).astype(np.uint8)
    hm = cv2.resize(hm, img.shape[:2], interpolation=cv2.INTER_LINEAR)
    return draw_heatmap(img, hm)

def predict(data_loader, net, encoder, config):
    net.eval()
    nes = []
    for i, (imgs, heatmaps, vismaps, kpts) in tqdm(enumerate(data_loader)):
        data = Variable(imgs.cuda(async=True), volatile=True)
        hm_preds1, hm_preds2 = net(data)
        hm_preds2 = F.relu(hm_preds2, False)
        for j in range(len(imgs)):
            img = imgs[j]
            img = np.transpose(img.numpy(), (1, 2, 0))
            img = ((img * config.sigma + config.mu) * 255).astype(np.uint8)
            hm_pred = hm_preds2[j].data.cpu().numpy()
            heatmap = heatmaps[j].numpy()
            x, y = encoder.decode_np(hm_pred, scale=1, stride=config.hm_stride, method='exp')
            keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
            kpt = kpts[j].numpy()
            ne = normalized_error(keypoints, kpt, img.shape[1])
            nes.append(ne)
            kp_img = draw_keypoints(img, keypoints, kpt)
            hps = []
            for k in range(len(hm_pred)):
                hp = hm_pred[k]
                hp2 = heatmap[k]
                hp = print_heatmap(hp, img, config)
                hp2 = print_heatmap(hp2, img, config)
                hps.append(np.concatenate([hp, hp2], axis=0))
            cv2.imwrite('/home/storage/lsy/fashion/tmp/%d-%d-hm.png' % (i, j), np.concatenate(hps, axis=1))
            cv2.imwrite('/home/storage/lsy/fashion/tmp/%d-%d-kp.png' % (i, j), kp_img)
    print(np.nanmean(np.array(nes)))





if __name__ == '__main__':
    batch_size = 32
    workers = 16
    config = Config('outwear')
    n_gpu = pytorch_utils.setgpu('0,1')
    test_kpda = KPDA(config, db_path, 'val')
    print('Test sample number: %d' % test_kpda.size())

    net = CascadePyramidNet(config) # UNet(config)  #VGG_FCN(config, layer_num=8) #ResidualUNet2D(config)  #
    checkpoint = torch.load(root_path + 'checkpoints/outwear_063_cascade.ckpt')  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)

    test_dataset = DataGenerator(config, test_kpda, phase='test')
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            collate_fn=test_dataset.collate_fn,
                            pin_memory=True)
    encoder = KeypointEncoder()
    predict(test_loader, net, encoder, config)