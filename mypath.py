class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'G:\\CV\\Reading\\DSSD\\data\\VOC2012'
        elif dataset == 'coco':
            return ''
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
    
    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return 'G:\\CV\\weights\\resnet101.pth'
        elif backbone == 'vgg16':
            return 'G:\\CV\\weights\\vgg16.pth'
        else:
            print('weights {} not available.'.format(backbone))
            raise NotImplementedError