import os
import visdom
import os.path as osp
import numpy as np
from tqdm import tqdm

from dataloaders.Detection_Dataset import Detection_Dataset
from model.DSSD import DSSD
from model.multibox_loss import MultiBoxLoss
from utils.config import cfg
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.visualization import create_vis_plot, update_vis_plot, model_info
from utils.saver import Saver
from utils.timer import Timer
from mypath import Path
from eval import Evaluator

import torch
import torch.utils.data as data
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg
        self.time = Timer()

        # Define Saver
        self.saver = Saver(args, cfg)
        self.saver.save_experiment_config()

        # Define Dataloader
        train_dataset = Detection_Dataset(args, cfg, 'train', 'train')
        self.num_classes = train_dataset.num_classes
        self.input_size = train_dataset.input_size
        self.train_loader = data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                num_workers=self.args.workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                collate_fn=train_dataset.collate_fn,
                drop_last=True)

        # Define Network
        # initilize the network here.
        if args.net == 'resnet':
            model = DSSD(cfg=cfg,
                         net=args.net,
                         output_stride=32,
                         num_classes=self.num_classes,
                         img_size=self.input_size,
                         pretrained=True,
                         mode='train')
        else:
            NotImplementedError

        train_params = [{'params': model.get_1x_lr_params(), 'lr': cfg.lr},
                        {'params': model.get_10x_lr_params(), 'lr': cfg.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params,
                                    momentum=cfg.lr,
                                    weight_decay=cfg.weight_decay,
                                    nesterov=False)

        # Define Criterion
        # Whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset))
            if osp.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset,
                                                  self.train_loader,
                                                  cfg.num_classes)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = MultiBoxLoss(args, cfg, self.num_classes, weight)
        self.model, self.optimizer = model, optimizer

        # Define lr scherduler
        self.scheduler = LR_Scheduler(args.lr_scheduler,
                                      cfg.lr,
                                      args.epochs,
                                      len(self.train_loader))

        # Resuming Checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Using cuda
        if args.ng > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=args.device_ids)
        self.model = self.model.to(args.device)

        # Clear start epoch if fine-tuning
        if args.ft:
            self.start_epoch = 0
        else:
            self.start_epoch = args.start_epoch

        # Visdom
        if args.visdom:
            vis = visdom.Visdom()
            vis_legend = ['Loss_local', 'Loss_confidence', 'mAP_50_F1', 'mAP_50_P', 'mAP_50_R']
            self.epoch_plot = create_vis_plot(vis, 'Epoch', 'Loss', 'train loss', vis_legend[0:2])
            self.batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', vis_legend[0:2])
            self.test_plot = create_vis_plot(vis, 'Epoch', 'Loss', 'test loss', vis_legend)
            self.vis = vis
            self.vis_legend = vis_legend
        model_info(self.model)

    def training(self, epoch):
        self.time.epoch()
        self.model.train()
        ave_loss_l = 0.
        ave_loss_c = 0.
        for ii, (images, targets, _, _) in enumerate(self.train_loader):
            self.time.batch()
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)
            self.scheduler(self.optimizer, ii, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(images)

            loss_l, loss_c = self.criterion(output, targets)
            loss = loss_l + loss_c
            ave_loss_c += (loss_c - ave_loss_c) / (ii + 1)
            ave_loss_l += (loss_c - ave_loss_l) / (ii + 1)
            assert torch.isnan(loss), 'WARNING: nan loss detected, ending training'
            loss.backward()
            self.optimizer.step()

            if self.args.visdom:
                update_vis_plot(self.vis, ii, [loss_l, loss_c], self.batch_plot, 'append')

            print('[Epoch: [%d], loc_loss: %10.3g, conf_loss: %10.3g, time: %5.2gs]' % (
                epoch, loss_l, loss_c, self.time.batch))

        print('[Epoch: [%d], average_loc_loss: %10.3g, average_conf_loss: %10.3g, time: %5.2gm]' % (
                epoch, ave_loss_l, ave_loss_c, self.time.epoch))


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    evaluator = Evaluator(args)
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.validate == (args.validate-1):
            evaluator.validation(trainer.model, epoch)

        if trainer.args.no_val:
            # save checkpoint every epoch
            is_best = False
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if trainer.args.ng > 1 else trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'best_pred': trainer.best_pred,
            }, is_best)

        if trainer.new_pred > trainer.best_pred:
            is_best = True
            trainer.best_pred = trainer.new_pred
            trainer.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trainer.model.module.state_dict() if trainer.args.ng > 1 else trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'best_pred': trainer.best_pred,
            }, is_best)


if __name__ == "__main__":
    main()
