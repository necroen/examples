import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def _flip_box(boxes, width):
    boxes = boxes.clone()
    oldx1 = boxes[:, 0].clone()
    oldx2 = boxes[:, 2].clone()
    boxes[:, 0] = width - oldx2 - 1
    boxes[:, 2] = width - oldx1 - 1
    return boxes


class TransformVOCDetectionAnnotation(object):
    def __init__(self, class_to_ind, keep_difficult=False):
        self.keep_difficult = keep_difficult
        self.class_to_ind = class_to_ind

    def __call__(self, target):
        boxes = []
        gt_classes = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            
            bb = obj.find('bndbox')
            bndbox = map(int, [bb.find('xmin').text, bb.find('ymin').text,
                               bb.find('xmax').text, bb.find('ymax').text])
            # xmin ymin xmax ymax   bndbox每一列的含义
            bndbox = list(bndbox)  # scj
            boxes += [bndbox]
            gt_classes += [self.class_to_ind[name]]

        size = target.find('size')
        im_info = map(int, (size.find('height').text, size.find('width').text, 3))

        res = {
            'boxes': torch.LongTensor(boxes),
            'gt_classes': gt_classes,
            'im_info': im_info
        }
        return res


class VOCSegmentation(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id)  # .convert('RGB')

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007' # root + dataset_name 所在目录就是 5 个文件夹并列的地方
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join( self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]
#        print('ids:', self.ids[index])
        target = ET.parse(self._annopath % img_id).getroot()    # 得到一个xml文件
        img = Image.open(self._imgpath % img_id).convert('RGB') # 得到图像
        
        if self.transform is not None:
            img = self.transform(img) # 对图像做 transform

        if self.target_transform is not None:
            target = self.target_transform(target) # 提取xml文件中的 bndbox 信息

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        bbox = target['boxes'].numpy() # n行 4列   xmin ymin xmax ymax
        indx = target['gt_classes']    # n个数字   类别对应的数字
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1)
        for i in range(bbox.shape[0]):
            bndbox = list(bbox[i,:])
            x = bndbox[0]
            y = bndbox[1]
            w = bndbox[2] - bndbox[0]
            h = bndbox[3] - bndbox[1]
            rect = patches.Rectangle( (x,y),w,h, linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            name = ind_to_class[ indx[i] ]
            ax.text(x-5, y-5, name, style='italic', color='red', fontsize=12)
        ax.imshow(img)
        


cls = ('__background__',  # always index 0
       'aeroplane', 'bicycle', 'bird', 'boat',
       'bottle', 'bus', 'car', 'cat', 'chair',
       'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'pottedplant',
       'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip( cls            , range(len(cls))  )) # 类别到数字的对应
ind_to_class = dict(zip( range(len(cls)), cls              )) # 数字到类别的对应
    


if __name__ == '__main__':
    dataset = VOCDetection('D:\\tfrcnn\VOC\VOCtrainval_06-Nov-2007\VOCdevkit', 'train',
                      target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))

    img, target = dataset[1]
    dataset.show(1398) # 随便显示一张图片
