'''Encode object boxes and labels.'''
import math
import torch

from utils import bbox_iou, bbox_nms


class DataEncoder:
    def __init__(self, config):
        self.config = config
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.
        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.config.anchor_areas:
            for ar in self.config.aspect_ratios:  # s = ar*h * h
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.config.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.config.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.
        Args:
          input_size: (tensor) model input size of (w,h).
        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.config.anchor_areas)
        fm_sizes = [(input_size/pow(2., i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xx = torch.arange(0, fm_w).repeat(fm_h).view(-1, 1)
            yy = torch.arange(0, fm_h).view(-1, 1).repeat(1, fm_w).view(-1, 1)
            xy = torch.cat([xx, yy], 1)  # anchor coordinates on feature map [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, self.config.anchor_num, 2)  # anchor coordinates on image
            wh = self.anchor_wh[i].view(1, 1, self.config.anchor_num, 2).expand(fm_h, fm_w, self.config.anchor_num, 2)
            box = torch.cat([xy,wh], 3)  # [fm_h, fm_w, num_anchor, 4] whose last dimension is [x,y,w,h]
            boxes.append(box.view(-1, 4))  # [x,y,w,h]
        return torch.cat(boxes, 0)

    def encode(self, bboxes, labels, input_size):
        '''Encode target bounding boxes and class labels.
        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        Args:
          bboxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          reg_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor(input_size)
        anchor_bboxes = self._get_anchor_boxes(input_size)
        # (xc, yc, w, h) -> (x1, y1, x2, y2)
        a = anchor_bboxes[:, :2]
        b = anchor_bboxes[:, 2:]
        anchor_bboxes_wh = torch.cat([a-b/2, a+b/2], 1)  # [anchor# 4]

        ious = bbox_iou(anchor_bboxes_wh, bboxes)  # [anchor#, object#] iou for each anchor and bbox
        max_ious, max_ids = ious.max(1)  # max (and indice) for each row (anchor)
        bboxes = bboxes[max_ids]

        # (x1, y1, x2, y2) -> (xc, yc, w, h)
        a = bboxes[:, :2]
        b = bboxes[:, 2:]
        bboxes = torch.cat([(a+b)/2, b-a+1], 1)  # [anchor# 4]
        loc_xy = (bboxes[:, :2] - anchor_bboxes[:, :2]) / anchor_bboxes[:, 2:]
        loc_wh = torch.log(bboxes[:, 2:] / anchor_bboxes[:, 2:])
        reg_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = labels[max_ids]
        cls_targets[max_ious < self.config.max_iou] = 0
        cls_targets[(max_ious > self.config.min_iou) & (max_ious < self.config.max_iou)] = -1  # for now just mark ignored to -1
        _, best_anchor_ids = ious.max(0)
        cls_targets[best_anchor_ids] = labels
        return reg_targets, cls_targets

    def decode(self, reg_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          reg_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, ].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = self.config.cls_thresh
        NMS_THRESH = self.config.nms_thresh

        input_size = torch.Tensor(input_size)
        anchor_bboxes = self._get_anchor_boxes(input_size)
        reg_xy = reg_preds[:, :2]
        reg_wh = reg_preds[:, 2:]
        xy = reg_xy * anchor_bboxes[:,2:] + anchor_bboxes[:,:2]
        wh = reg_wh.exp() * anchor_bboxes[:,2:]
        bboxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]
        score = cls_preds.sigmoid()  # [#anchors,]
        ids = score > CLS_THRESH

        if ids.long().sum() == 0:
            _, ids = ids.max(0)
        else:
            ids = ids.nonzero().squeeze()  # index for true
        keep = bbox_nms(bboxes[ids], score[ids], threshold=NMS_THRESH, topk=self.config.nms_topk)
        return bboxes[ids][keep], score[ids][keep]

