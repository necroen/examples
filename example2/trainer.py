from collections import namedtuple
# http://blog.csdn.net/helei001/article/details/52692128

import time
from torch.nn import functional as F

from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at

#from utils.vis_tool import Visualizer

from utils.config import opt

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """
    FasterRCNN训练器，返回losses
    
    losses 包括
    rpn_loc_loss: rpn位置误差
    rpn_cls_loss: rpn分类误差
    roi_loc_loss: roihead 或者说 Fast R-CNN 位置误差
    roi_cls_loss: roihead 或者说 Fast R-CNN 分类误差
    total_loss: 以上 4 种误差的总和

    传入参数:一个将要被训练的 FasterRCNN 模型
    """
    
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
#        训练RPN的时候，从20000个anchor中选择256个进行训练，
#        以使得正负样本比例大概是1:1. 同时给出训练的 位置参数目标。
        
        self.proposal_target_creator = ProposalTargetCreator()
#        在训练RoIHead/Fast R-CNN的时候，从 2000 个 rois 中选择 128 个用以训练。
#        同时给定训练目标, 返回（sample_RoI, gt_RoI_loc, gt_RoI_label）

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std
        self.optimizer = self.faster_rcnn.get_optimizer()
        

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        
        # 20000->12000->2000 rois 
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        
        # ProposalTargetCreator 
        # 在训练RoIHead/Fast R-CNN的时候，从 2000 个 rois 中选择 128 个用以训练。
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, sample_roi_index)
        
        # ------------------ RPN losses -------------------#
#        AnchorTargetCreator
#        训练RPN的时候，从20000个anchor中选择256个进行训练，
#        以使得正负样本比例大概是1:1. 同时给出训练的 位置参数目标。
        
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator( at.tonumpy(bbox), anchor, img_size)
        
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1) 

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long(), at.totensor(gt_roi_label).long()]
        
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label )

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        
        # 没有显示平均误差
        print('rpnLoc:',  float(rpn_loc_loss.data.numpy()), ' rpnCls:', float(rpn_cls_loss.data.numpy()), 
              ' roiLoc:', float(rpn_loc_loss.data.numpy()), ' roiCls:', float(roi_cls_loss.data.numpy()) )
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            
            import os
#            import stat
            save_path = os.getcwd() + '\\checkpoints\\fasterrcnn_%s' % timestr
#            print('save_path:', save_path)
            os.makedirs(save_path)
#            os.chmod(save_path, stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
#            os.chmod(save_path, stat.S_IRWXO)
            
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        t.save(save_dict, save_path) # permission denied  这里还要处理下
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum()  # ignore gt_label==-1 for rpn_loss
    return loc_loss
