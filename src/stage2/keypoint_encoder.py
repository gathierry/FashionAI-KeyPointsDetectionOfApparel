import torch
import numpy as np
import torch.nn.functional as F
import cv2

class KeypointEncoder:

    def _gaussian_keypoint(self, input_size, mu_x, mu_y, alpha, sigma):
        h, w = input_size
        x = torch.linspace(0, w-1, steps=w)
        y = torch.linspace(0, h-1, steps=h)
        xx = x.repeat(w, 1)
        yy = y.repeat(h, 1).t()
        zz = alpha * torch.exp(-( ((xx-mu_x)**2) / (2*sigma**2) + ((yy-mu_y)**2) / (2*sigma**2) ))
        return zz

    def _gaussian_keypoint_np(self, input_size, mu_x, mu_y, alpha, sigma):
        h, w = input_size
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        xx, yy = np.meshgrid(x, y)
        zz = alpha * np.exp(-( ((xx-mu_x)**2) / (2*sigma**2) + ((yy-mu_y)**2) / (2*sigma**2) ))
        return zz

    def encode(self, keypoints, input_size, stride, hm_alpha, hm_sigma):
        '''
        :param keypoints: [pt num, 3] -> [[x, y, vis], ...]
        :return: [pt num, h, w] [h, w]
        '''
        kpt_num = len(keypoints)
        vismap = torch.zeros([kpt_num, ])
        inps = [x//stride for x in input_size]
        kpts = keypoints.clone()
        kpts[:,:2] = (kpts[:,:2] - 1.0) / stride
        h, w = inps
        heatmap = torch.zeros([kpt_num, h, w])
        for c, kpt in enumerate(kpts):
            x, y, v = kpt
            if v >= 0:
            #if v == 1 and x > 0 and y > 0:
                heatmap[c] = self._gaussian_keypoint(inps, x, y, hm_alpha, hm_sigma)
            vismap[c] = v
        return heatmap, vismap

    # def encode_np(self, keypoints, input_size, stride, hm_alpha, hm_sigma):
    #     '''
    #     :param keypoints: [pt num, 3] -> [[x, y, vis], ...]
    #     :return: [pt num, h, w] [h, w]
    #     '''
    #     kpt_num = len(keypoints)
    #     vismap = np.zeros([kpt_num, ], dtype=np.float32)
    #     kpts = np.copy(keypoints)
    #     inps = [x // stride for x in input_size]
    #     h, w = inps
    #     heatmap = np.zeros([kpt_num, h, w], dtype=np.float32)
    #     for c, kpt in enumerate(kpts):
    #         x, y, v = kpt
    #         if v == 1 and x > 0 and y > 0:
    #             heatmap[c] = self._gaussian_keypoint_np(inps, float(x-1)/stride, float(y-1)/stride, hm_alpha, hm_sigma)
    #         vismap[c] = v
    #     return heatmap, vismap

    def decode_np(self, heatmap, scale, stride, default_pt, method='exp'):
        '''
        :param heatmap: [pt_num, h, w]
        :param scale:
        :return: 
        '''
        kp_num, h, w = heatmap.shape
        dfx, dfy = np.array(default_pt) * scale / stride
        for k, hm in enumerate(heatmap):
            heatmap[k] = cv2.GaussianBlur(hm, (5, 5), 1)
        if method == 'exp':
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            heatmap_th = np.copy(heatmap)
            heatmap_th[heatmap<np.amax(heatmap)/2] = 0
            heat_sums_th = np.sum(heatmap_th, axis=(1, 2))
            x = np.sum(heatmap_th * xx, axis=(1, 2))
            y = np.sum(heatmap_th * yy, axis=(1, 2))
            x = x / heat_sums_th
            y = y / heat_sums_th
            x[heat_sums_th == 0] = dfx
            y[heat_sums_th == 0] = dfy
        else:
            if method == 'max':
                heatmap_th = heatmap.reshape(kp_num, -1)
                y, x = np.unravel_index(np.argmax(heatmap_th, axis=1), [h, w])
            elif method == 'maxoffset':
                heatmap_th = heatmap.reshape(kp_num, -1)
                si = np.argsort(heatmap_th, axis=1)
                y1, x1 = np.unravel_index(si[:, -1], [h, w])
                y2, x2 = np.unravel_index(si[:, -2], [h, w])
                x = (3 * x1 + x2) / 4.
                y = (3 * y1 + y2) / 4.
            var = np.var(heatmap_th, axis=1)
            x[var<1] = dfx
            y[var<1] = dfy
        x = x * stride / scale
        y = y * stride / scale
        return np.rint(x+2), np.rint(y+2)



if __name__ == '__main__':
    from kpda_parser import KPDA
    import cv2
    from src.config import Config
    import numpy as np
    config = Config()
    db_path = '/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/train/'
    kpda = KPDA(db_path, 'train')
    img_path = kpda.get_image_path(2)
    kpts = kpda.get_keypoints(2)
    kpts = torch.from_numpy(kpts)
    img = cv2.imread(img_path)
    image = np.zeros([512, 512, 3])
    image[:512, :504, :] = img
    cv2.imwrite('/home/storage/lsy/fashion/tmp/img.jpg', image)
    ke = KeypointEncoder()
    heatmaps, _ = ke.encode(kpts, image.shape[:2], config.hm_stride)
    for i, heatmap in enumerate(heatmaps):
        heatmap = np.expand_dims(heatmap.numpy() * 255, 2)
        cv2.imwrite('/home/storage/lsy/fashion/tmp/map%d.jpg' % i, heatmap )