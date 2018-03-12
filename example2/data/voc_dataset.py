import os
import xml.etree.ElementTree as ET

import numpy as np

from data.util import read_image


class VOCBboxDataset:
    """
    y_{min}, x_{min}, y_{max}, x_{max}
    ymin xmin ymax xmax
    
    这个 data_dir 是那个 5 个文件夹并列的路径
    Annotations   ImageSets   JPEGImages   SegmentationClass   SegmentationObject
    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.
        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based  注意这里减了 1
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')]) # ymin xmin ymax xmax
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


if __name__ == '__main__':
    
    # 这里实现了同时画 2 张图片 及 标注
    data_dir = 'D:\\tfrcnn'+ '\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    dataset = VOCBboxDataset(data_dir)
    
    i = 12
    img_1, bbox_1, label_1, _ = dataset[i]
    img_1 = img_1.astype(np.uint8)
    img_1 = img_1.transpose((1, 2, 0)) 
    
    i = 1024
    img_2, bbox_2, label_2, _ = dataset[i]
    img_2 = img_2.astype(np.uint8)
    img_2 = img_2.transpose((1, 2, 0)) 
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig = plt.figure()
    #=============================
    ax1 = fig.add_subplot(1, 2, 1)  # ax1 type  AxesSubplot
    
    for i in range(bbox_1.shape[0]):  
        bndbox = list(bbox_1[i,:])
        # ymin xmin ymax xmax
        #    0    1    2    3
        x = bndbox[1]  # xmin
        y = bndbox[0]  # ymin
        w = bndbox[3] - bndbox[1]  # xmax - xmin 
        h = bndbox[2] - bndbox[0]  # ymax - ymin
        rect = patches.Rectangle( (x,y),w,h, linewidth=2,edgecolor='r',facecolor='none')
        ax1.add_patch(rect)
        name = VOC_BBOX_LABEL_NAMES[ label_1[i] ]
        ax1.text(x-5, y-5, name, style='italic', color='red', fontsize=12)
        
    ax1.imshow(img_1)
    #=============================
    ax2 = fig.add_subplot(1, 2, 2)  # ax2 type  AxesSubplot
    
    for i in range(bbox_2.shape[0]):
        bndbox = list(bbox_2[i,:])
        x = bndbox[1]
        y = bndbox[0]
        w = bndbox[3] - bndbox[1]
        h = bndbox[2] - bndbox[0]
        
        rect = patches.Rectangle( (x,y),w,h, linewidth=2,edgecolor='r',facecolor='none')
        ax2.add_patch(rect)
        
        name = VOC_BBOX_LABEL_NAMES[ label_2[i] ]
        ax2.text(x-5, y-5, name, style='italic', color='red', fontsize=12)
        
    ax2.imshow(img_2)
