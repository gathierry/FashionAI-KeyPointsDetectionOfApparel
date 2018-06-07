import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def focal_loss(self, output, labels):
        pt = output * labels + (1 - output) * (1 - labels)
        return torch.mean(-self.alpha * (1-pt)**self.gamma * torch.log(pt+1e-7))

    def forward(self, reg_preds, reg_targets, cls_preds, cls_targets):
        '''
        :param reg_preds: [batch_size, anchor#, 4]
        :param reg_targets: [batch_size, anchor#, 4]
        :param cls_preds: [batch_size, anchor#, class#]
        :param cls_targets: [batch_size, anchor#]
        :return: 
        '''
        batch_size = cls_targets.size(0)
        pos = cls_targets > 0.5  # [batch_size, anchor#]
        num_pos = pos.data.long().sum()

        # regression loss
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(reg_preds)  # [N, anchor#] -> [N, anchor#, 1] -> [N, anchor#, 4]
            masked_reg_preds = reg_preds[mask].view(-1, 4)  # [pos#, 4]
            masked_reg_targets = reg_targets[mask].view(-1, 4)  # [pos#, 4]
            regress_loss = F.smooth_l1_loss(masked_reg_preds, masked_reg_targets, size_average=True)
        else:
            regress_loss = 0

        # classification loss
        neg = (cls_targets > -0.5) & (cls_targets < 0.5)
        num_neg = neg.data.long().sum()
        mask_pos = pos.unsqueeze(2).expand_as(cls_preds)
        mask_neg = neg.unsqueeze(2).expand_as(cls_preds)
        masked_pos_cls_preds = F.sigmoid(cls_preds[mask_pos])
        masked_neg_cls_preds = F.sigmoid(cls_preds[mask_neg])

        classify_loss = 0.5*self.focal_loss(masked_pos_cls_preds, cls_targets[pos]) \
                        + 0.5*self.focal_loss(masked_neg_cls_preds, cls_targets[neg])
        loss = classify_loss + regress_loss
        pos_total = num_pos
        neg_total = num_neg
        pos_correct = ((masked_pos_cls_preds > 0.5) * (cls_targets[pos] > 0.5)).data.long().sum()
        neg_correct = ((masked_neg_cls_preds < 0.5) * (cls_targets[neg] < 0.5)).data.long().sum()
        return [loss, classify_loss, regress_loss] + [pos_correct, pos_total, neg_correct, neg_total]


