import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from stage1.label_encoder import DataEncoder


class DataGenerator(Dataset):
    def __init__(self, config, data, phase='train'):
        self.phase = phase
        self.data = data
        self.config = config
        self.encoder = DataEncoder(self.config)

    def __getitem__(self, idx):
        img = cv2.imread(self.data.get_image_path(idx))  # BGR
        bboxes = self.data.get_bbox(idx)
        img_h, img_w, _ = img.shape
        # data augmentation
        if self.phase == 'train':
            random_flip = np.random.randint(0, 2)
            if random_flip == 1:
                img = cv2.flip(img, 1)
                x1s = img_w - bboxes[:, 2]
                x2s = img_w - bboxes[:, 0]
                bboxes[:, 0] = x1s
                bboxes[:, 2] = x2s
        # min size resizing
        scale = self.config.img_max_size / max(img_w, img_h)
        img_h2 = int(img_h * scale)
        img_w2 = int(img_w * scale)
        img = cv2.resize(img, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
        bboxes *= scale
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
        img[[0, 2]] = img[[2, 0]]
        img = img / 255.0
        img = (img - self.config.mu) / self.config.sigma
        return torch.from_numpy(img), torch.from_numpy(bboxes)

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        bboxes = [x[1] for x in batch]
        # Use the same size to accelerate dynamic graph
        maxh = self.config.img_max_size #max([img.size(1) for img in imgs])
        maxw = self.config.img_max_size #max([img.size(2) for img in imgs])
        num_imgs = len(imgs)
        pad_imgs = torch.zeros(num_imgs, 3, maxh, maxw)
        reg_targets = []
        cls_targets = []
        for i in range(num_imgs):
            img = imgs[i]
            pad_imgs[i, :, :img.size(1), :img.size(2)] = img  # Pad images to the same size
            reg_target, cls_target = self.encoder.encode(bboxes[i], torch.ones([1,]), [maxh, maxw])
            reg_targets.append(reg_target)
            cls_targets.append(cls_target)
        reg_targets = torch.stack(reg_targets)  # [batch_size, anchor#, 4]
        cls_targets = torch.stack(cls_targets)  # [batch_size, anchor#] 0 for neg, 1, 2, 3 ... for different classes
        return pad_imgs, reg_targets, cls_targets

    def __len__(self):
        return self.data.size()


if __name__ == '__main__':
    from config import Config
    from coco import Coco
    from torch.utils.data import DataLoader
    from time import time
    db_path = '/home/storage/lsy/coco/'

    config = Config()
    train_coco = Coco(db_path, 'train')

    train_dataset = DataGenerator(config, train_coco, phase='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=16,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    t0 = time()
    for i, (data, reg_targets, cls_targets) in enumerate(train_loader):
        print(data.size(), reg_targets.size(), cls_targets.size())
    t1 = time()
