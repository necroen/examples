import torch.nn as nn
from roi_pooling import roi_pooling

from rpn import RPN
from faster_rcnn import FasterRCNN

import torch
import torchvision.models as models

sqz = models.squeezenet1_0(pretrained=False)
sqz.load_state_dict(torch.load('squeezenet1_0-a815701f.pth'))
sqz_feature_extractor = nn.Sequential( *list(sqz.features.children()) )

class Feature_extractor(nn.Module):
    def __init__(self, feature_extractor):
        super(Feature_extractor, self).__init__()
        self.m = feature_extractor

    def forward(self, x):
        return self.m(x)
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.m1 = nn.Linear(512 * 7 * 7, 21)
        self.m2 = nn.Linear(512 * 7 * 7, 21 * 4)

    def forward(self, x):
        return self.m1(x), self.m2(x)


def pooler(x, rois, spatial_scale_H, spatial_scale_W):
    x = roi_pooling(x, rois, spatial_scale_H, spatial_scale_W, size=(7, 7))
    return x.view(x.size(0), -1)


from torch.nn import functional as F

class RPNClassifier(nn.Module):
    def __init__(self, n):
        super(RPNClassifier, self).__init__()
#        self.conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # 转义
        self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
        self.m2 = nn.Conv2d(n, 36, 3, 1, 1)
        
#        normal_init(self.conv1, 0, 0.01)
        normal_init(self.m1, 0, 0.01)
        normal_init(self.m2, 0, 0.01)

    def forward(self, x):
#        x = F.relu(self.conv1(x))
        return self.m1(x), self.m2(x)


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


FasterRCNN_model = FasterRCNN(
    features = Feature_extractor(sqz_feature_extractor),
    pooler = pooler,
    classifier = Classifier(),
    rpn = RPN( classifier = RPNClassifier(512) )
)
