import os

#import ipdb
#from IPython.core.debugger import Pdb
#ipdb = Pdb()


#import matplotlib
#from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer

import torch

from utils import array_tool as at

from utils.vis_tool import visdom_bbox
#from utils.eval_tool import eval_detection_voc



def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    # img, bbox, label, scale = dataset[0]
    # 返回的img是被scale后的图像，可能已经被随机翻转了
    # 返回的 bbox 按照 ymin xmin ymax xmax 排列
    #  H, W = size(im)
    # 对于一张屏幕上显示的图片，a,b,c,d 代表 4 个顶点 
    #        a   ...   b     ymin
    #        .         .
    #        c   ...   d     ymax  H高度    y的范围在 [0, H-1] 间
    #        xmin    xmax
    #        W宽度   x的范围在 [0, W-1] 间
    
    print('load data')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn)
    
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
        
        
    for epoch in range(opt.epoch):
        for ii, (img, bbox_, label_, scale) in (enumerate(dataloader)):
            print('step: ', ii)
            
            scale = at.scalar(scale)
            img, bbox, label = img.float(), bbox_, label_
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)
            
            if ( (ii + 1) % opt.plot_every == 0 ) and (epoch > 50): 
            # 运行多少步以后再predict一次，epoch跑的太少的话根本预测不准什么东西
#                if os.path.exists(opt.debug_file):
#                    ipdb.set_trace()

                # plot groud truth bboxes  画出原本的框
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # gt_img  np类型  范围是 [0, 1] 间   3 x H x W
                # 这里要将 gt_img 这个带框，带标注的图像保存或者显示出来

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # 预测结果显示在pred_img上， 也应该保存或者显示出来


if __name__ == '__main__':
    train()
