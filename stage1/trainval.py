import os
import time

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
from capb_parser import CAPB
from config import Config
from stage1.data_generator import DataGenerator
from stage1.retinanet import RetinaNet
from stage1.focal_loss import FocalLoss
from lr_scheduler import LRScheduler


root_path = '/home/storage/lsy/fashion/'
db_path = root_path + 'Category_and_Attribution_Prediction_Benchmark/'


def print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir, log_mode):
    str0 = 'Epoch %03d (lr %.5f)' % (epoch, lr)
    str1 = 'Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(train_metrics[:, 3]) / np.sum(train_metrics[:, 4]),
        100.0 * np.sum(train_metrics[:, 5]) / np.sum(train_metrics[:, 6]),
        np.sum(train_metrics[:, 4]), np.sum(train_metrics[:, 6]), train_time)
    str2 = 'loss %2.4f, classify loss %2.4f, regress loss %2.4f' % (
        np.mean(train_metrics[:, 0]), np.mean(train_metrics[:, 1]), np.mean(train_metrics[:, 2]))
    str3 = 'Validation: tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(val_metrics[:, 3]) / np.sum(val_metrics[:, 4]),
        100.0 * np.sum(val_metrics[:, 5]) / np.sum(val_metrics[:, 6]),
        np.sum(val_metrics[:, 4]), np.sum(val_metrics[:, 6]), val_time)
    str4 = 'loss %2.4f, classify loss %2.4f, regress loss %2.4f' % (
        np.mean(val_metrics[:, 0]), np.mean(val_metrics[:, 1]), np.mean(val_metrics[:, 2]))

    print(str0)
    print(str1)
    print(str2)
    print(str3)
    print(str4 + '\n')
    if epoch > 1:
        log_mode = 'a'
    f = open(save_dir + 'rpn_train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str1 + '\n')
    f.write(str2 + '\n')
    f.write(str3 + '\n')
    f.write(str4 + '\n\n')
    f.close()

def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, reg_targets, cls_targets) in tqdm(enumerate(data_loader)):
        data = Variable(data.cuda(async=True))
        reg_targets = Variable(reg_targets.cuda(async=True))
        cls_targets = Variable(cls_targets.cuda(async=True))
        reg_preds, cls_preds = net(data)
        loss_output = loss(reg_preds, reg_targets, cls_preds, cls_targets)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()
        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    return metrics, end_time - start_time


def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    for i, (data, reg_targets, cls_targets) in tqdm(enumerate(data_loader)):
        data = Variable(data.cuda(async=True), volatile=True)
        reg_targets = Variable(reg_targets.cuda(async=True), volatile=True)
        cls_targets = Variable(cls_targets.cuda(async=True), volatile=True)
        reg_preds, cls_preds = net(data)
        loss_output = loss(reg_preds, reg_targets, cls_preds, cls_targets)
        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


if __name__ == '__main__':
    batch_size = 32
    workers = 16
    n_gpu = pytorch_utils.setgpu('all')
    epochs = 100
    base_lr = 1e-3
    save_dir = root_path + 'checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    resume = True

    config = Config('outwear')  # param doesn't matter here
    net = RetinaNet(config, num_classes=2)
    loss = FocalLoss()

    train_data = CAPB(db_path, 1, 'train')  # "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes
    val_data = CAPB(db_path, 1, 'val')
    print('Train sample number: %d' % train_data.size())
    print('Val sample number: %d' % val_data.size())

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if resume:
        checkpoint = torch.load(save_dir + 'rpn_006.ckpt')
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
        if val_loss < best_val_loss or epoch % 10 == 0 or lr is None:
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
                os.path.join(save_dir, 'rpn_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break


