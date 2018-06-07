import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from src.stage2.keypoint_encoder import KeypointEncoder


class DataGenerator(Dataset):
    def __init__(self, config, data, phase='train'):
        self.phase = phase
        self.data = data
        self.config = config
        self.encoder = KeypointEncoder()

    def __getitem__(self, idx):
        img = cv2.imread(self.data.get_image_path(idx))  # BGR
        kpts = self.data.get_keypoints(idx)
        img_h, img_w, _ = img.shape
        # data augmentation
        if self.phase == 'train':
            random_flip = np.random.randint(0, 2)
            if random_flip == 1:
                img = cv2.flip(img, 1)
                kpts[:, 0] = img_w - kpts[:, 0]
                for conj in self.config.conjug:
                    kpts[conj] = kpts[conj[::-1]]
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img_w, img_h), flags=cv2.INTER_CUBIC)
            kpts_tmp = kpts.copy()
            kpts_tmp[:, 2] = 1
            kpts[:, :2] = np.matmul(kpts_tmp, M.T)
        # min size resizing
        scale = self.config.img_max_size / max(img_w, img_h)
        img_h2 = int(img_h * scale)
        img_w2 = int(img_w * scale)
        img = cv2.resize(img, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
        kpts[:, :2] = kpts[:, :2] * scale
        kpts = kpts.astype(np.float32)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
        img[[0, 2]] = img[[2, 0]]
        img = img / 255.0
        img = (img - self.config.mu) / self.config.sigma

        # transfer from collate_fn
        # maxh = self.config.img_max_size  # max([img.size(1) for img in imgs])
        # maxw = self.config.img_max_size  # max([img.size(2) for img in imgs])
        # pad_img = np.zeros([3, maxh, maxw], dtype=np.float32)
        # pad_img[:, :img.shape[1], :img.shape[2]] = img
        # heatmap, vismap = self.encoder.encode_np(kpts, [maxh, maxw], self.config.hm_stride,
        #                                          self.config.hm_alpha, self.config.hm_sigma)
        # return torch.from_numpy(pad_img), torch.from_numpy(heatmap), torch.from_numpy(vismap)
        return torch.from_numpy(img), torch.from_numpy(kpts)

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        kpts = [x[1] for x in batch]
        # Use the same size to accelerate dynamic graph
        maxh = self.config.img_max_size #max([img.size(1) for img in imgs])
        maxw = self.config.img_max_size #max([img.size(2) for img in imgs])
        num_imgs = len(imgs)
        pad_imgs = torch.zeros(num_imgs, 3, maxh, maxw)
        heatmaps = []
        vismaps = []
        for i in range(num_imgs):
            img = imgs[i]
            pad_imgs[i, :, :img.size(1), :img.size(2)] = img  # Pad images to the same size
            heatmap, vismap = self.encoder.encode(kpts[i], [maxh, maxw], self.config.hm_stride,
                                                  self.config.hm_alpha, self.config.hm_sigma)
            heatmaps.append(heatmap)
            vismaps.append(vismap)
            # vismaps.append((vismap>0).float())
        heatmaps = torch.stack(heatmaps)  # [batch_size, kptnum, h, w]
        vismaps = torch.stack(vismaps)  # [batch_size, kptnum]
        if self.phase == 'test':
            return pad_imgs, heatmaps, vismaps, kpts
        return pad_imgs, heatmaps, vismaps

    def __len__(self):
        return self.data.size()


if __name__ == '__main__':
    from src.config import Config
    from src.kpda_parser import KPDA
    from torch.utils.data import DataLoader


    config = Config('dress')
    train_data = KPDA(config, config.data_path, 'train')

    data_dataset = DataGenerator(config, train_data, phase='train')
    data_loader = DataLoader(data_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              collate_fn=data_dataset.collate_fn,
                              pin_memory=True)
    for i, (imgs, heatmaps, vismaps) in enumerate(data_loader):
        print()
        # for j in range(len(imgs)):
        #     img = imgs[j]
        #     img = np.transpose(img.numpy(), (1, 2, 0))
        #     img = ((img * config.sigma + config.mu) * 255).astype(np.uint8)
        #     hm_pred = heatmaps[j].numpy()
        #     for k in range(len(hm_pred)):
        #         hp = hm_pred[k]
        #         hp_max = np.amax(hp)
        #         scale = 0
        #         if hp_max != 0:
        #             scale = 255//hp_max
        #         hp = (hp * scale).astype(np.uint8)
        #         hp = cv2.resize(hp, img.shape[:2], interpolation = cv2.INTER_LINEAR)
        #         draw = draw_heatmap(img, hp)
        #         cv2.imwrite('/home/storage/lsy/fashion/tmp/%d-%d-%d.png' % (i, j, k), draw)
        # if i == 0:
        #     break
