from pprint import pprint



class Config:
    # 这个 data_dir 是那个 5 个文件夹并列的路径
    voc_data_dir  = 'D:\\tfrcnn\\VOC\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007'
    test_data_dir = 'D:\\tfrcnn\\VOC\\VOCtest_06-Nov-2007\\VOCdevkit\\VOC2007'
    min_size = 600    # dataset 把这个min_size 当做最小size，如果图片有哪一个边比这个还小，则按比例scale
    max_size = 1000   # 没用到
    num_workers =      0  # 原来是8
    test_num_workers = 0  # 原来是8
    
    rpn_sigma = 3.  # sigma for l1_smooth_loss
    roi_sigma = 1.

    # param for optimizer
    weight_decay = 0.0005  # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    lr_decay = 0.1         # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter
    
    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    epoch = 10  # training

    use_adam =    False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop =    False  # use dropout in RoIHead
    
    debug_file = '/tmp/debugf'  # debug
    test_num = 10000
    
    pretrainend_model_path = 'D:\\tfrcnn\simple2\model\squeezenet1_0-a815701f.pth'  # model
    
    load_path = None
    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
 

opt = Config()