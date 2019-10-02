class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/twsf/work/DSSD/data/VOC2012'
        elif dataset == 'coco':
            return '/home/twsf/data/COCO'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

    def weights_root_dir(backbone):
        if backbone == 'resnet101':
            return '/home/twsf/.cache/torch/checkpoints/resnet101.pth'
        else:
            print('weights {} not available.'.format(backbone))
            raise NotImplementedError
