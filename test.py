"""
data aug methods
"""
import cv2
from tqdm import tqdm
import numpy as np

from model.DSSD import DSSD
from utils.config import cfg
from utils.timer import Timer

import torch
import torch.utils.data as data
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Test(object):
    def __init__(self, args, work_dir):
        self.args = args
        self.cfg = cfg
        self.time = Timer()
        self.input_size = cfg.input_size
        self.num_classes = 21
        self.class_name = ('__background__',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor')
        # Define Network
        # initilize the network here.
        if args.net == 'resnet':
            model = DSSD(args=args,
                         cfg=cfg,
                         net=args.net,
                         output_stride=32,
                         num_classes=self.num_classes,
                         img_size=self.input_size,
                         pretrained=True)
        else:
            NotImplementedError
        checkpoint = torch.load(work_dir)
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model.to(self.args.device)

    def test(self, img_path):
        self.time.batch()
        self.model.eval()
        image = cv2.imread(img_path)
        img, ratio, left, top = self.letterbox(image, self.input_size)
        img2 = img
        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = self.normalize(img)

        # image transform to input form of network
        img = torch.from_numpy(img).unsqueeze(0)
        input = img.to(self.args.device)

        output = self.model(input)

        output = output.squeeze(0).cpu()
        output = output[output[:, 4].gt(0)]
        output[:, :4] *= self.input_size
        for ii, name in enumerate(self.class_name):
            print(ii, ':', name, end='. ')
        print(output)
        # output[:, [0, 2]] = output[:, [0, 2]] / ratio - left
        # output[:, [1, 3]] = output[:, [1, 3]] / ratio - top
        # self.show_image(image, output)
        self.show_image(img2, output)

        print("Time cost: %7.3gs" % self.time.batch())

    def letterbox(self, img, height=512, color=(127.5, 127.5, 127.5)):
        """resize a rectangular image to a padded square
        """
        shape = img.shape[:2]  # shape = [height, width]
        ratio = float(height) / max(shape)  # ratio  = old / new
        dw = (max(shape) - shape[1]) / 2  # width padding
        dh = (max(shape) - shape[0]) / 2  # height padding
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        interp = np.random.randint(0, 5)
        img = cv2.resize(img, (height, height), interpolation=interp)  # resized, no border

        return img, ratio, left, top

    def normalize(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """norm = (x - mean) / std
        """
        img = img / 255.0
        mean = np.array(mean)
        std = np.array(std)
        img = (img - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
        return img.astype(np.float32)

    def show_image(self, img, labels):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1).imshow(img[:, :, ::-1])
        plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '-')
        plt.show()
        pass


def main():
    import os
    import os.path as osp
    from utils.hyp import parse_args
    args = parse_args()

    test = Test(args, '/home/twsf/work/DSSD/work_dirs/pascal/dssd-resnet/model_best.pth.tar')
    root = '/home/twsf/data/VOC2012/JPEGImages'
    img_list = os.listdir(root)
    for img in img_list:
        path = osp.join(root, img)
        test.test(path)


if __name__ == "__main__":
    main()
