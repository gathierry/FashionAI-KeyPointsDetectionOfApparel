import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import pandas as pd
import argparse
import sys

from src import pytorch_utils
from src.config import Config
from src.kpda_parser import KPDA
from src.stage2.cascade_pyramid_network import CascadePyramidNet
from src.stage2v9.cascade_pyramid_network_v9 import CascadePyramidNetV9
from src.utils import draw_heatmap, draw_keypoints
from src.stage2.keypoint_encoder import KeypointEncoder


def data_frame_template():
    df = pd.DataFrame(columns=['image_id','image_category','neckline_left','neckline_right','center_front','shoulder_left',
                               'shoulder_right','armpit_left','armpit_right','waistline_left','waistline_right',
                               'cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left',
                               'top_hem_right','waistband_left','waistband_right','hemline_left','hemline_right',
                               'crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out'])
    return df

def compute_keypoints(config, img0, net, encoder, doflip=False):
    img_h, img_w, _ = img0.shape
    # min size resizing
    scale = config.img_max_size / max(img_w, img_h)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)
    img = cv2.resize(img0, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
    img[[0, 2]] = img[[2, 0]]
    img = img / 255.0
    img = (img - config.mu) / config.sigma
    pad_imgs = np.zeros([1, 3, config.img_max_size, config.img_max_size], dtype=np.float32)
    pad_imgs[0, :, :img_h2, :img_w2] = img
    data = torch.from_numpy(pad_imgs)
    data = Variable(data.cuda(async=True), volatile=True)
    _, hm_pred = net(data)
    hm_pred = F.relu(hm_pred, False)
    hm_pred = hm_pred[0].data.cpu().numpy()
    if doflip:
        a = np.zeros_like(hm_pred)
        a[:, :, :img_w2 // config.hm_stride] = np.flip(hm_pred[:, :, :img_w2 // config.hm_stride], 2)
        for conj in config.conjug:
            a[conj] = a[conj[::-1]]
        hm_pred = a
    return hm_pred

    # x, y = encoder.decode_np(hm_pred, scale, config.hm_stride, method='maxoffset')
    # keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
    # return keypoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--clothes', help='specify the clothing type', default='trousers')
    parser.add_argument('-g', '--gpu', help='cuda device to use', default='0')
    parser.add_argument('-m1', '--model1', help='specify the model', default=None)
    parser.add_argument('-m2', '--model2', help='specify the model', default=None)
    parser.add_argument('-v', '--visual', help='whether visualize result', default=False)
    args = parser.parse_args(sys.argv[1:])

    config = Config(args.clothes)
    n_gpu = pytorch_utils.setgpu(args.gpu)
    test_kpda = KPDA(config, config.data_path, 'test')
    print('Testing ' + config.clothes)
    print('Test sample number: %d' % test_kpda.size())
    df = data_frame_template()
    cudnn.benchmark = True
    net = CascadePyramidNet(config)
    checkpoint = torch.load(args.model1)  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    net = DataParallel(net)
    net.eval()
    net2 = CascadePyramidNetV9(config)
    checkpoint = torch.load(args.model1)  # must before cuda
    net2.load_state_dict(checkpoint['state_dict'])
    net2 = net2.cuda()
    net2 = DataParallel(net2)
    net2.eval()
    encoder = KeypointEncoder()
    for idx in tqdm(range(test_kpda.size())):
        img_path = test_kpda.get_image_path(idx)
        img0 = cv2.imread(img_path)  # BGR
        img0_flip = cv2.flip(img0, 1)

        img_h, img_w, _ = img0.shape
        scale = config.img_max_size / max(img_w, img_h)
        hm_pred = compute_keypoints(config, img0, net, encoder)
        hm_pred_flip = compute_keypoints(config, img0_flip, net, encoder, doflip=True)
        hm_pred2 = compute_keypoints(config, img0, net2, encoder)
        hm_pred_flip2 = compute_keypoints(config, img0_flip, net2, encoder, doflip=True)
        x, y = encoder.decode_np(hm_pred + hm_pred_flip + hm_pred2 + hm_pred_flip2,
                                 scale, config.hm_stride, (img_w/2, img_h/2), method='maxoffset')
        keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)

        # keypoints = compute_keypoints(config, img0, net, encoder)
        # keypoints_flip = compute_keypoints(config, img0_flip, net, encoder)
        # keypoints_flip[:, 0] = img0.shape[1] - keypoints_flip[:, 0]
        # for conj in config.conjug:
        #     keypoints_flip[conj] = keypoints_flip[conj[::-1]]
        # keypoints[:, :2] = (keypoints[:, :2] + keypoints_flip[:, :2]) // 2

        row = test_kpda.anno_df.iloc[idx]
        df.at[idx, 'image_id'] = row['image_id']
        df.at[idx, 'image_category'] = row['image_category']
        for k, kpt_name in enumerate(config.keypoints[config.clothes]):
            df.at[idx, kpt_name] = str(keypoints[k,0])+'_'+str(keypoints[k,1])+'_1'

        if args.visual:
            kp_img = draw_keypoints(img0, keypoints)
            cv2.imwrite('/home/storage/lsy/fashion/tmp/%d.png' % idx, kp_img)
    df.fillna('-1_-1_-1', inplace=True)
    print(df.head(5))
    df.to_csv(config.proj_path +'kp_predictions/'+config.clothes+'.csv', index=False)
