import torch as t
from torch import nn
#from torchvision.models import vgg16

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN

#from model.roi_module import RoIPooling2D
from model.roi_pooling import roi_pooling

from utils import array_tool as at
from utils.config import opt


#def decom_vgg16():
#    # the 30th layer of features is relu of conv5_3
#    if opt.caffe_pretrain:
#        model = vgg16(pretrained=False)
#        if not opt.load_path:
#            model.load_state_dict(t.load(opt.caffe_pretrain_path))
#    else:
#        model = vgg16(not opt.load_path)
#
#    features = list(model.features)[:30]
#    classifier = model.classifier
#
#    classifier = list(classifier)
#    del classifier[6]
#    if not opt.use_drop:
#        del classifier[5]
#        del classifier[2]
#    classifier = nn.Sequential(*classifier)
#
#    # freeze top4 conv
#    for layer in features[:10]:
#        for p in layer.parameters():
#            p.requires_grad = False
#
#    return nn.Sequential(*features), classifier


def decom_sqz():
    ''' 用 squeezeNet 前面提取feature的部分替换掉 vgg16 的feature提取部分'''
    import torch
    import torchvision.models as models
    import torch.nn as nn
    
    sqz = models.squeezenet1_0(pretrained=False)
    sqz.load_state_dict(torch.load(opt.pretrainend_model_path))
    features = nn.Sequential( *list(sqz.features.children()) )
    
    classifier = nn.Sequential(
                  nn.Linear(25088, 4096),
                  nn.ReLU(inplace = True),
                  nn.Linear(4096,  4096),
                  nn.ReLU(inplace = True)
                 )
    
    return features, classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class    = 20,
                 ratios        = [0.5, 1, 2],
                 anchor_scales = [8, 16, 32]
                 ):
                 
#        extractor, classifier = decom_vgg16()
        extractor, classifier = decom_sqz()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    """
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
#        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.
        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        # 0 ymin xmin ymax xmax
        # 0    1    2    3    4
        # 变成
        # 0 xmin ymin xmax ymax   所以是 yx -> xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        
        
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

#        pool = self.roi(x, indices_and_rois) # 这里做的是roi_pooling
        pool = roi_pooling(x, indices_and_rois, self.spatial_scale)
        #                     Variable
        
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
