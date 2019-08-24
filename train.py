import os
import visdom
import os.path as osp
import numpy as np

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
        train_dataset = Detection_Dataset(args, cfg, cfg.train_split, 'train')
        self.num_classes = train_dataset.num_classes
        self.input_size = train_dataset.input_size
        self.train_loader = data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                num_workers=self.args.workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                collate_fn=train_dataset.collate_fn)
        self.num_batch = len(self.train_loader)

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
        # self.optimizer = self.model.to(self.args.device)
        self.model = self.model.to(self.args.device)
        if args.ng > 1 and args.use_multi_gpu:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=args.gpu_ids)
        # Clear start epoch if fine-tuning
        if args.ft:
            self.start_epoch = 0
        else:
            self.start_epoch = args.start_epoch

        # Visdom
        if args.visdom:
            vis = visdom.Visdom()
            vis_legend = ['Loss_local', 'Loss_confidence', 'mAP', 'mF1']
            self.epoch_plot = create_vis_plot(vis, 'Epoch', 'Loss', 'train loss', vis_legend[0:2])
            self.batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', vis_legend[0:2])
            self.val_plot = create_vis_plot(vis, 'Epoch', 'result', 'val loss', vis_legend[2:4])
            self.vis = vis
            self.vis_legend = vis_legend
        model_info(self.model)

    def training(self, epoch):
        self.time.epoch()
        self.model.train()
        ave_loss_l = 0.
        ave_loss_c = 0.
        for ii, (images, targets, _, _) in enumerate(self.train_loader):
            num_target = [len(ann) for ann in targets]
            # continue if exist image no target.
            if 0 in num_target:
                continue
            self.time.batch()
            images = images.to(self.args.device)
            targets = [ann.to(self.args.device) for ann in targets]
            self.scheduler(self.optimizer, ii, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(images, mode='train')

            loss_l, loss_c = self.criterion(output, targets)
            loss = loss_l + loss_c
            ave_loss_c += (loss_c - ave_loss_c) / (ii + 1)
            ave_loss_l += (loss_l - ave_loss_l) / (ii + 1)
            assert not torch.isnan(loss), 'WARNING: nan loss detected, ending training'
            loss.backward()
            self.optimizer.step()

            # visdom
            if self.args.visdom:
                update_vis_plot(self.vis, ii, [loss_l, loss_c], self.batch_plot, 'append')

            show_info = '[mode: train' +\
                'Epoch: [%d][%d/%d], ' % (epoch, ii, self.num_batch) +\
                'lr: %5.4g, ' % self.optimizer.param_groups[0]['lr'] +\
                'loc_loss: %5.3g, conf_loss: %5.3g, time: %5.2gs]' %\
                (loss_l, loss_c, self.time.batch())
            if (ii + 1) % 50 == 0:
                print(show_info)

            # Save log info
            self.saver.save_log(show_info)

        epoch_show_info = '[mode: train, ' +\
            'Epoch: [%d], ' % epoch +\
            'lr: %5.4g, ' % self.optimizer.param_groups[0]['lr'] +\
            'average_loc_loss: %5.3g, ' % ave_loss_l +\
            'average_conf_loss: %5.3g, ' % ave_loss_c +\
            'time: %5.2gm]' % self.time.epoch()
        print(epoch_show_info)

        # Save log info
        self.saver.save_log(epoch_show_info)

        # visdom
        if self.args.visdom:
            update_vis_plot(self.vis, epoch, [ave_loss_l, ave_loss_c], self.epoch_plot, 'append')


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    evaluator = Evaluator(args)
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val and epoch % args.validate == (args.validate - 1):
            map, mf1 = evaluator.validation(trainer.model, epoch)
            val_svar_pf = '[mode: val ' +\
                'mAP: %5.4g, ' % map +\
                'mF1: %5.4g]' % mf1
            trainer.saver.save_log(val_svar_pf)
            if args.visdom:
                update_vis_plot(trainer.vis, epoch, [map, mf1], trainer.val_plot, 'append')

        if evaluator.is_best:
            trainer.best_pred = evaluator.new_pred

        if args.is_save:
            # save checkpoint every epoch
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() \
                    if args.ng > 1 and args.use_multi_gpu else trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'best_pred': evaluator.best_pred,
            }, evaluator.is_best)


if __name__ == "__main__":
    main()
