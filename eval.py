from tqdm import tqdm
import numpy as np

from dataloaders.Detection_Dataset import Detection_Dataset
from model.multibox_eval import MultiBoxEval
from model.DSSD import DSSD
from utils.config import cfg
from utils.timer import Timer

import torch
import torch.utils.data as data
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg
        self.time = Timer()
        self.batch_size = args.eval_batch_size
        self.best_pred = 0
        self.is_best = False

        # Define Dataloader
        val_dataset = Detection_Dataset(args, cfg, 'val', 'val')
        self.num_classes = val_dataset.num_classes
        self.classes = val_dataset.classes
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
            model = DSSD(args=args,
                         cfg=cfg,
                         net=args.net,
                         output_stride=32,
                         num_classes=self.num_classes,
                         img_size=self.cfg.input_size,
                         pretrained=True)
        else:
            NotImplementedError
        if self.args.eval_from is not None:
            model.load_state_dict(torch.load(self.args.eval_from))
            self.model = model.to(self.args.device)
            if args.ng > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=args.gpu_ids)

        # Define evalutor
        self.evalutor = MultiBoxEval(self.cfg.input_size, self.cfg.iou_thresh)
        self.evalutor = self.evalutor.to(self.args.device)
        if args.ng > 1:
            self.evalutor = torch.nn.DataParallel(self.evalutor, device_ids=args.gpu_ids)

    def validation(self, model=None, epoch=None):
        self.time.total()
        if model is not None:
            self.model = model
        assert self.model is not None
        self.model.eval()

        # (correct, conf, pcls, tcls)
        stats = []
        num_img = 0

        tbar = tqdm(self.val_loader, desc='\r')
        for ii, (images, targets, _, _) in enumerate(tbar):
            images = images.to(self.args.device)
            targets = [ann.to(self.args.device) for ann in targets]
            bs = images.shape[0]
            num_img += bs

            with torch.no_grad():
                output = self.model(images, mode='val')
            stats += self.evalutor(output, targets, bs)

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=self.num_classes)
        if len(stats):
            p, r, ap, f1, ap_class = self.ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        # Print results
        pf = "[mode: 'val', epoch: [%d], num_img: %6d, " +\
            "targets: %7d, precision: %3.3g, recall: %3.3g, " +\
            "mAP: %3.3g, F1: %3.3g]"  # print format
        print(pf % (epoch, num_img, nt.sum(), mp, mr, map, mf1))

        # Print results per class
        _pf = "[mode: 'val', class: %12s, targets: %7d, " +\
            "precision: %7.3g, recall: %7.3g, " +\
            "AP: %7.3g, F1: %7.3g]"  # print format
        if self.num_classes > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(_pf % (self.classes[c+1], nt[c], p[i],
                             r[i], ap[i], f1[i]))

        # self.new_pred
        self.new_pred = map
        if self.new_pred > self.best_pred:
            self.best_pred = self.new_pred
            self.is_best = True

        return map, mf1

    def ap_per_class(self, tp, conf, pred_cls, target_cls):
        """ Compute the average precision, given the recall and precision curves.
        # Arguments
            tp:    True positives (list).
            conf:  Objectness value from 0-1 (list).
            pred_cls: Predicted object classes (list).
            target_cls: True object classes (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        idx = np.argsort(-conf)
        tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]

        # Find unique classes
        unique_classes = np.unique(target_cls)

        # Create Precision-Recall curve and compute AP for each class
        ap, p, r = [], [], []
        for c in unique_classes:
            idx = (pred_cls == c)
            n_gt = (target_cls == c).sum()
            n_p = idx.sum()

            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[idx]).cumsum()
                tpc = (tp[idx]).cumsum()

                # Recall
                recall_curve = tpc / (n_gt + 1e-16)
                r.append(recall_curve[-1])

                # Precision
                precision_curve = tpc / (tpc + fpc)
                p.append(precision_curve[-1])

                # AP from recall-precision curve
                ap.append(self.compute_ap(recall_curve, precision_curve))

        # Compute F1 score (harmonic mean of precision and recall)
        p, r, ap = np.array(p), np.array(r), np.array(ap)
        f1 = 2 * p * r / (p + r + 1e-16)

        return p, r, ap, f1, unique_classes.astype('int32')

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Correct AP calculation
        # first append sentinel values at the end

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [1.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i-1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        idx = np.where(mrec[1:] != mrec[:-1])[0]

        # add sum (\Delta recall) * prec
        ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

        return ap


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Evaluator(args)
    trainer.validation()


if __name__ == "__main__":
    main()
