# -*- coding: utf-8 -*-
"""
http://www.cnblogs.com/shepherd2015/p/8535358.html
"""
import time
import torch

from torch.autograd import Variable
import numpy as np

from voc import VOCDetection, TransformVOCDetectionAnnotation
import torchvision.transforms as transforms
#%% 1. Hyper Parameters 
data_path = 'D:\\tfrcnn'+ '\VOC\VOCtrainval_06-Nov-2007\VOCdevkit'

#data_path = '/home/py_file/VOC/VOCtrainval_06-Nov-2007/VOCdevkit'
#%% 2. Dataset
from voc import class_to_ind, ind_to_class

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std =[0.229, 0.224, 0.225]),
                                ])
        
train_data = VOCDetection(data_path, 'train',
                          transform = transforms,
                          target_transform=\
                          TransformVOCDetectionAnnotation(class_to_ind, False))

def collate_fn(batch):
    imgs, gt = zip(*batch)
    return imgs[0].unsqueeze(0), gt[0]

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
#%% 3. Model
from pipeline import FasterRCNN_model
#%% 4. Loss and Optimizer
# Loss已经在模型中定义了
import torch.optim as optim

learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4

optimizer = optim.SGD(FasterRCNN_model.parameters(), lr = learning_rate,
                      momentum = momentum,
                      weight_decay = weight_decay)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
#%% 5. Train the Model
from utils import AverageMeter
import globalvar as gl

for epoch in range(90):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    rpn_cls_losses = AverageMeter()
    rpn_reg_losses = AverageMeter()
    fst_cls_losses = AverageMeter()
    fst_reg_losses = AverageMeter()
    
    FasterRCNN_model.train()
    end = time.time()
    
    for i, (im, gt) in (enumerate(train_loader)):
        adjust_learning_rate(optimizer, epoch)
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        loss, scores, boxes = FasterRCNN_model((im, gt))
        loss.backward()
        optimizer.step()
        
        losses.update(loss.data[0], im.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        rpn_cls_loss = gl.get_value('rpn_cls_loss')
        rpn_reg_loss = gl.get_value('rpn_reg_loss')
        fst_cls_loss = gl.get_value('fst_cls_loss')
        fst_reg_loss = gl.get_value('fst_reg_loss')
        
        rpn_cls_losses.update(float( rpn_cls_loss ))
        rpn_reg_losses.update(float( rpn_reg_loss ))
        fst_cls_losses.update(float( fst_cls_loss ))
        fst_reg_losses.update(float( fst_reg_loss ))
        
        print('Epoch: [{0}][{1:4d}]  '
              'Time {batch_time.avg:.3f}  '
              'Loss {loss.val:.4f} ({loss.avg:.6f})  '
              'rpncls {rpnv1:.6f}  '
              'rpnreg {rpnv2:.6f}  '
              'fstcls {fstv1:.6f}  '
              'fstreg {fstv2:.6f}  '
              .format(
                      epoch, i, batch_time=batch_time,
                      loss=losses,
                      rpnv1 = rpn_cls_losses.avg,
                      rpnv2 = rpn_reg_losses.avg,
                      fstv1 = fst_cls_losses.avg,
                      fstv2 = fst_reg_losses.avg
              ))
#%% 6 Save the Model
torch.save(FasterRCNN_model.state_dict(), 'FasterRCNN_model.pkl')
#%% 7. Test the Model
# pass
#%% 8 测试单张图像
# pass