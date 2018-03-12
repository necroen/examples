from __future__ import division
import torch as t
import numpy as np

#import cupy as cp

from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
#from model.utils.nms import non_maximum_suppression
from model.utils.py_cpu_nms import py_cpu_nms

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.
    """

    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. 
        These values are a threshold value used for non maximum suppression 
        and a threshold value to discard low confidence proposals in :meth:`predict`,
        respectively.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7   # 只保留预测出来的概率大于 0.7 的框
            
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05  # 只保留概率大于 0.05 的框
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class 略过背景，只做了20类
        for l in range(1, self.n_class): # 对每一个类别逐一做 nms 
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            
#            print('prob_l size:', prob_l.shape )  # 在 predict 的时候 rpn提供 300个框给roi head
            print('prob_l:\n', prob_l ) # 都是非常小的数
            # 在这里，不满足条件的话就全部给扔了，下面都不用跑了
            
            mask = prob_l > self.score_thresh  # discard low confidence proposals in `predict`
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
#            keep = non_maximum_suppression(cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            # 注意要替换掉
#            print('cls_bbox_l size:', cls_bbox_l.shape ) # 空的
            print('prob_l     size:', prob_l.shape )     # 空的
            
            prob_l = prob_l[:, np.newaxis]
            dets = np.hstack((cls_bbox_l, prob_l))
            keep = py_cpu_nms(dets, self.nms_thresh) # list
            
            print('keep number:', len(keep) )
#            keep = cp.asnumpy(keep)
            keep = np.array(keep)
            
            
            # 当整个网络预测能力还很差的、没有训练好的时候，没有一个框的预测概率能大于 score_thresh = 0.7 
            # 全部是无效的预测框，然后这里keep的数目就为 0，下面这句就可能会报错
            bbox.append(cls_bbox_l[keep])  
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def predict(self, imgs,sizes=None,visualize=False):
        """Detect objects from images.
        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: (R, 4)
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
           * **labels** : 
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : 
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')   # 本来是 visualize   evaluate
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale) # 这里调用了 forward 方法
            
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1)) # 有趣 打出来的分，softmax后变成概率值
            # 可以在这里看一下预测出来的最大概率是多少，如果太小就直接return出去，下面都不用跑了
            # prob 是 300 x 21 的尺寸， np.sum(prob) = 300

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
                    
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer