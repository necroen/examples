# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:31:04 2018

@author: Administrator
"""
import numpy as np

def py_cpu_nms(dets, thresh):
    """
    non maximum suppression
    非极大值抑制
    http://blog.csdn.net/lin_xiaoyi/article/details/78858990
    """
#    x1 = dets[:, 0] # xmin
#    y1 = dets[:, 1] # ymin
#    x2 = dets[:, 2] # xmax
#    y2 = dets[:, 3] # ymax
#    scores = dets[:, 4]
    # dets 默认有5列 ，按照 xmin ymin xmax ymax score 的顺序排列
    # 如果不是上面的情况的话，需要做合适的修改
    
#          ymin xmin ymax xmax
#          0    1    2    3
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # xmax - xmin    ymax - ymin
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