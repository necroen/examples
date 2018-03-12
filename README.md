### <p align="center">examples</p>  
两个cpu版本的 Faster R-CNN，都是基于pytorch的。一如既往，目的还是为了学习。。。  
当然是修改的别人的实现  
全部改成了用 squeezeNet 1.0 做特征提取器，不保证最终效果  
只要下两个东西，一个squeezeNet 1.0 的 [pth 文件](https://download.pytorch.org/models/squeezenet1_0-a815701f.pth)，一个 VOC2007 的数据集，我在Anaconda3里头跑的    
作为初学者，我只想要一个摁一下 run 按钮就能跑，不要配置一堆玩意儿的版本，两个版本中 example2 比 example1 完善的多，但是学习的话建议先看 1 再看 2，两个最好都从处理数据集的部分开始看。  

**example1** 改的 [fmassa/fast_rcnn](https://github.com/pytorch/examples/tree/d8d378c31d2766009db400ac03f41dd837a56c2a/fast_rcnn)，修了下各种小bug，只有train，没有 predict 部分，anchor的4列采用的是**xmin ymin xmax ymax**顺序，入口在main.py    
  
**example2** 改的 [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)，有 predict 部分，删了cupy部分，删了visdom部分，删了计算mAP的部分，删了cuda版的 nms 和 roipooling，全部用example1里头的 nms 和 roipooling替代，删了使用 caffe 预训练模型的部分，修改参数只能在utils/config.py里做，anchor的4列采用的是**ymin xmin ymax xmax**的顺序，入口在train.py  
  
Have fun!