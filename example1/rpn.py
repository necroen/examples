import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

# clean up environment
from utils import bbox_transform, bbox_transform_inv, clip_boxes, filter_boxes, bbox_overlaps
from utils import to_var as _tovar

from generate_anchors import generate_anchors

import globalvar as gl
gl._init()
#%%
def nms(dets, thresh):
    """
    non maximum suppression
    非极大值抑制
    http://blog.csdn.net/lin_xiaoyi/article/details/78858990
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
#%%
class RPN(nn.Module):
    def __init__(self,
                 classifier, 
                 anchor_scales=None,
                 negative_overlap=0.3, 
                 positive_overlap=0.7,
                 fg_fraction=0.5, 
                 batch_size=256,
                 nms_thresh=0.7, 
                 min_size=16,
                 pre_nms_topN =12000, 
                 post_nms_topN=2000
                 ):
        super(RPN, self).__init__()

        self.rpn_classifier = classifier

        if anchor_scales is None:
            anchor_scales = (8, 16, 32)
        self._anchors = generate_anchors(scales=np.array(anchor_scales)) # 9 x 4 的anchor矩阵，机械过程
        self._num_anchors = self._anchors.shape[0] # 9

        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap
        self.fg_fraction = fg_fraction
        self.batch_size = batch_size

        # used for both train and test
        self.nms_thresh = nms_thresh
        self.pre_nms_topN = pre_nms_topN
        self.post_nms_topN = post_nms_topN
        self.min_size = min_size

    # output rpn probs as well
    def forward(self, im, feats, gt=None):
        assert im.size(0) == 1, 'only single element batches supported'
        
        self._feat_stride = round(im.size(3) / feats.size(3)) # 如果不round的话，只会约等于 16

        rpn_map, rpn_bbox_pred = self.rpn_classifier(feats) 
        # 根据feats,算出每个框的前后景分数，算出每个框的四个坐标
        
        all_anchors = self.rpn_get_anchors(feats)
        
        rpn_loss = None
        if gt is not None:    # if self.training is True:
            assert gt is not None
            # 挑出 256 个 sample 来计算 loss。 rpn_labels 中等于1和0的数字刚好有256个，其他都是-1，代表不关心
            rpn_labels, rpn_bbox_targets = self.rpn_targets(all_anchors, im, gt)
            
            rpn_loss = self.rpn_loss(rpn_map, rpn_bbox_pred, rpn_labels, rpn_bbox_targets)

        # clip, sort, pre nms topk, nms, after nms topk
        # params are different for train and test
        # proposal_layer.py
        # 挑出2000个框提供给 fast rcnn 用，scores 是相应的前景打分
        roi_boxes, scores = self.get_roi_boxes(all_anchors, rpn_map, rpn_bbox_pred, im)

        return _tovar((roi_boxes, scores, rpn_loss))


    def rpn_get_anchors(self, feats):   
        # 对feats上的每个点都生成9个anchor，并reshape成 n x 4 的形式，这个函数比较机械
        # n = height x width x 9
        height, width = feats.size()[-2:] # 得到feats的高height和宽width
        
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        A = self._num_anchors # 9
        K = shifts.shape[0]   # height x width
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    def rpn_targets(self, all_anchors, im, gt):
        # 挑出 256 个框来计算 loss
        total_anchors = all_anchors.shape[0]
        gt_boxes = gt['boxes']

        height, width = im.size()[-2:]
        # only keep anchors inside the image
        _allowed_border = 0
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < width + _allowed_border) &   # width
            (all_anchors[:, 3] < height + _allowed_border)    # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
#        print('anchors inside:', anchors.shape[0])
        assert anchors.shape[0] > 0, '{0}x{1} -> {2}'.format(height, width, total_anchors)
        # 这一行可能会报错！！！  报错是因为有些图片很窄  96 x 500， 即便最窄的框anchor都比图片某条边宽一点

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        # overlaps = bbox_overlaps(anchors, gt_boxes)#.numpy()
        overlaps = bbox_overlaps(torch.from_numpy(anchors), gt_boxes).numpy()
        gt_boxes = gt_boxes.numpy()
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self.negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.positive_overlap] = 1

        # subsample positive labels if we have too many
        num_fg = int(self.fg_fraction * self.batch_size)
        fg_inds = np.where(labels == 1)[0]
        
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.batch_size - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        #bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        #bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])

        # map up to original set of anchors
        labels       = _unmap(labels,       total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

        return labels, bbox_targets

    # I need to know the original image size (or have the scaling factor)
    def get_roi_boxes(self, anchors, rpn_map, rpn_bbox_deltas, im):
        # TODO fix this!!!
        im_info = (100, 100, 1)

        bbox_deltas = rpn_bbox_deltas.data.numpy()
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        #scores = bottom[0].data[:, self._num_anchors:, :, :]
        scores = rpn_map.data[:, self._num_anchors:, :, :].numpy() # 这里其实是只取了前景的分数
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im.size()[-2:])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = filter_boxes(proposals, self.min_size * im_info[2]) # scj
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        
        keep = nms(np.hstack((proposals, scores)), self.nms_thresh)  # nms!!!
        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        
        proposals = proposals[keep, :]
        scores = scores[keep]

        return proposals, scores

    def rpn_loss(self, rpn_map, rpn_bbox_transform, rpn_labels, rpn_bbox_targets):
        height, width = rpn_map.size()[-2:]

        rpn_map = rpn_map.view(-1, 2, height, width).permute(0,2, 3, 1).contiguous().view(-1, 2)
        
        labels = torch.from_numpy(rpn_labels).long()  # convert properly
        labels = labels.view(1, height, width, -1).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(-1)

        idx = labels.ge(0).nonzero()[:, 0]
        rpn_map = rpn_map.index_select(0, Variable(idx, requires_grad=False))
        labels = labels.index_select(0, idx)
        labels = Variable(labels, requires_grad=False)

        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets)
        rpn_bbox_targets = rpn_bbox_targets.view(1, height, width, -1).permute(0, 3, 1, 2)
        rpn_bbox_targets = Variable(rpn_bbox_targets, requires_grad=False)

        cls_crit = nn.CrossEntropyLoss()
        reg_crit = nn.SmoothL1Loss()
        cls_loss = cls_crit(rpn_map, labels)
        # verify normalization and sigma
        reg_loss = reg_crit(rpn_bbox_transform, rpn_bbox_targets)

        gl.set_value('rpn_cls_loss', cls_loss.data.numpy() )
        gl.set_value('rpn_reg_loss', reg_loss.data.numpy() )

        loss = cls_loss + reg_loss
        return loss
    

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
