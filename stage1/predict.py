import numpy as np
import torch
from data_generator import DataGenerator
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import pytorch_utils
from config import Config
from kpda_parser import KPDA
from stage1.label_encoder import DataEncoder
from stage1.retinanet import RetinaNet
from utils import draw_bbox

root_path = '/home/storage/lsy/fashion/'
db_path = root_path + 'FashionAI_Keypoint_Detection/'

def predict(data_loader, net, config):
    net.eval()
    for i, (imgs, reg_targets, cls_targets) in enumerate(data_loader):
        data = Variable(imgs.cuda(async=True), volatile=True)
        reg_preds, cls_preds = net(data)
        encoder = DataEncoder(config)
        for j in range(len(imgs)):
            img = imgs[j]
            reg_pred = reg_preds.cpu()[j]
            cls_pred = cls_preds.cpu()[j]
            reg_target = reg_targets[j]
            cls_target = cls_targets[j]
            bboxes, scores = encoder.decode(reg_pred.data, cls_pred.data.squeeze(), [config.img_max_size, config.img_max_size])
            bboxes2, scores2 = encoder.decode(reg_target, cls_target, [config.img_max_size, config.img_max_size])
            img = np.transpose(img.numpy(), (1, 2, 0))
            img = ((img * config.sigma + config.mu) * 255).astype(np.uint8)
            draw_bbox(img, bboxes.numpy(), scores.numpy(), '/home/storage/lsy/fashion/predictions/'+config.clothes+'/%d-%d.png' % (i, j), bboxes2.numpy())




if __name__ == '__main__':
    batch_size = 24
    workers = 16
    config = Config('outwear')
    n_gpu = pytorch_utils.setgpu('all')
    test_kpda = KPDA(config, db_path, 'val')
    print('Val sample number: %d' % test_kpda.size())


    net = RetinaNet(config, num_classes=2)
    checkpoint = torch.load(root_path + 'checkpoints/rpn_053.ckpt')  # must before cuda
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
    predict(test_loader, net, config)