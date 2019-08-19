import os
import os.path as osp
from tqdm import tqdm
import numpy as np

from dataloaders.Detection_Dataset import Detection_Dataset
from model.multibox_eval import MultiBoxEval
from model.DSSD import DSSD
from utils.config import cfg
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.timer import Timer
from box_utils import xywh_2_xyxy, jaccard

import torch
import torch.utils.data as data
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg
        self.time = Timer
        self.batch_size = args.eval_batch_size

        # Define Dataloader
        val_dataset = Detection_Dataset(args, cfg, 'val', 'val')
        self.num_classes = val_dataset.num_classes
        self.val_loader = data.DataLoader(
                val_dataset, batch_size=self.batch_size,
                num_workers=self.args.workers,
                shuffle=True,
                pin_memory=True,
                collate_fn=val_dataset.collate_fn,
                drop_last=True)
        self.val_num_images = len(self.val_loader)

        # Define Network
        # initilize the network here.
        self.model = None
        if args.net == 'resnet':
            model = DSSD(cfg=cfg,
                         net=args.net,
                         output_stride=32,
                         num_classes=self.num_classes,
                         img_size=self.input_size,
                         pretrained=True,
                         mode='val')
        else:
            NotImplementedError
        if self.args.eval_from is not None:
            model.load_state_dict(torch.load(self.args.eval_from))
            self.model = model.to(self.args.device)

        # Define evalutor
        self.evalutor = MultiBoxEval(self.cfg.input_size, self.cfg.iou_thresh)

    def validation(self, model=None, epoch=None):
        self.time.total()
        if model is not None:
            self.model = model
            self.model.mode = 'val'
        assert self.model is not None

        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        stats = []

        for ii, (images, targets, _, _) in enumerate(tbar):
            images = images.to(self.args.device)
            bs = images.shape[0]

            with torch.no_grad():
                output = self.model(images)

            stats += self.evalutor(output, targets, bs)

            
        # self.new_pred


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Evaluator(args)
    trainer.validation()


if __name__ == "__main__":
    main()
