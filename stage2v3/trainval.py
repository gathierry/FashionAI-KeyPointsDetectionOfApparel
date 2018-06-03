import os
import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/lsy/Fashion/')

import pytorch_utils
from kpda_parser import KPDA
from config import Config
from stage2.data_generator import DataGenerator
from stage2v3.cascade_pyramid_network_v3 import CascadePyramidNetV3
from stage2v3.viserrloss_v3 import VisErrorLossV3
from lr_scheduler import LRScheduler


root_path = '/home/storage/lsy/fashion/'
db_path = root_path + 'FashionAI_Keypoint_Detection/'


def print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir, log_mode):
    train_metrics = np.mean(train_metrics, axis=0)
    val_metrics = np.mean(val_metrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str1 = 'Train:      time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f loss3: %2.4f' \
           % (train_time, train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3])
    str2 = 'Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f loss3: %2.4f' \
           % (val_time, val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3])

    print(str0)
    print(str1)
    print(str2 + '\n')
    if epoch > 1:
        log_mode = 'a'
    f = open(save_dir + 'kpt_'+config.clothes+'_train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str1 + '\n')
    f.write(str2 + '\n\n')
    f.close()

def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        heatmaps = Variable(heatmaps.cuda(async=True))
        vismaps = Variable(vismaps.cuda(async=True))
        heat_pred1, heat_pred2 = net(data)
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()
        metrics.append([loss_output[0].data[0], loss_output[1].data[0],
                        loss_output[2].data[0], loss_output[3].data[0]])

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    return metrics, end_time - start_time


def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        data = Variable(data.cuda(async=True), volatile=True)
        heatmaps = Variable(heatmaps.cuda(async=True), volatile=True)
        vismaps = Variable(vismaps.cuda(async=True), volatile=True)
        heat_pred1, heat_pred2 = net(data)
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        metrics.append([loss_output[0].data[0], loss_output[1].data[0],
                        loss_output[2].data[0], loss_output[3].data[0]])
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--clothes', help='one cloth type', default='outwear')
    args = parser.parse_args(sys.argv[1:])
    print('Training ' + args.clothes)

    batch_size = 8
    workers = 4
    n_gpu = pytorch_utils.setgpu('all')
    epochs = 100
    base_lr = 1e-3 # SGD L1 loss starts from 1e-2, L2 loss starts from 1e-3
    save_dir = root_path + 'checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    resume = False

    config = Config(args.clothes)
    net = CascadePyramidNetV3(config)  #UNet(config) #VGG_FCN(config, layer_num=8) #ResidualUNet2D(config)  #
    loss = VisErrorLossV3()
    train_data = KPDA(config, db_path, 'train')
    val_data = KPDA(config, db_path, 'val')
    print('Train sample number: %d' % train_data.size())
    print('Val sample number: %d' % val_data.size())

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if resume:
        checkpoint = torch.load(save_dir + 'kpt_outwear_020.ckpt')
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'


    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)


    train_dataset = DataGenerator(config, train_data, phase='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    val_dataset = DataGenerator(config, val_data, phase='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            collate_fn=val_dataset.collate_fn,
                            pin_memory=True)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)

    lrs = LRScheduler(lr, epochs, patience=3, factor=0.1, min_lr=1e-5, early_stop=5, best_loss=best_val_loss)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_time = train(train_loader, net, loss, optimizer, lr)
        val_metrics, val_time = validate(val_loader, net, loss)

        print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir, log_mode)

        val_loss = np.mean(val_metrics[:, 0])
        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch%10 == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss},
                os.path.join(save_dir, 'kpt_'+config.clothes+'_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break


