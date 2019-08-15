import os
import time
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
from mypath import Path

import torch
import torch.utils.data as data
import multiprocessing
multiprocessing.set_start_method('spawn', True)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg

        # Define Saver
        self.saver = Saver(args, cfg)
        self.saver.save_experiment_config()

        # Define Dataloader
        train_dataset = Detection_Dataset(args, cfg, 'train', 'train')
        val_dataset = Detection_Dataset(args, cfg, 'val', 'val')
        self.num_classes = train_dataset.num_classes
        self.input_size = train_dataset.input_size
        self.train_loader = data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                num_workers=self.args.workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True)
        self.val_loader = data.DataLoader(
                val_dataset, batch_size=args.batch_size,
                num_workers=self.args.workers,
                shuffle=True,
                pin_memory=True,
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

        # Define Evaluater
        

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
        self.model.train()
        for ii, (image, target) in enumerate(self.train_loader):
            image.to(self.args.device)
            target.to(self.args.device)
            self.scheduler(self.optimizer, ii, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss_l, loss_c = self.criterion(output, target)
            loss = loss_l + loss_c
            assert torch.isnan(loss), 'WARNING: nan loss detected, ending training'
            loss.backward()
            self.optimizer.step()

            if self.args.visdom:
                update_vis_plot(self.vis, ii, [loss_l, loss_c], self.batch_plot, 'append')

            print('[Epoch: %d, numImages: %5d, loss: %10.3g]' % (
                epoch, ii * self.args.batch_size + image.data.shape[0], loss))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.args.ng > 1 else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        self.model.mode = 'val'
        for ii, (image, target) in enumerate(tbar):
            image, target = image.to(self.args.device), target.to(self.args.device) 
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %3.f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)            
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.args.ng > 1 else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == "__main__":
    main()