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
from mypath import Path
import torch


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg
        # Define Saver

        # Define Dataloader
        self.train_loader = Detection_Dataset(args, cfg, 'train', 'train')
        self.val_loader = Detection_Dataset(args, cfg, 'val', 'val')

        # Define Network
        # initilize the network here.
        if args.net == 'resnet':
            model = DSSD(cfg=cfg,
                         net=args.net,
                         output_stride=32,
                         num_classes=self.train_loader.num_classes,
                         img_size=self.train_loader.input_size,
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
        criterion = MultiBoxLoss(args, cfg, self.train_loader.num_classes, weight)
        self.model, self.optimizer = model, optimizer

        # Define Evaluater
        

        # Define lr scherduler
        self.scherduler = LR_Scheduler(args.lr_scheduler, 
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
            vis_legend = ['Loss', 'mAP_50', 'F1']
            self.epoch_plot = create_vis_plot(vis, 'Epoch', 'Loss', 'train loss', [vis_legend[0],])
            self.batch_plot = create_vis_plot(vis, 'Batch', 'Loss', 'batch loss', [vis_legend[0],])
            self.test_plot = create_vis_plot(vis, 'Epoch', 'Loss', 'test loss', vis_legend)
    
        model_info(self.model)


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval -  1):
            trainer.validation(epoch)
    
    trainer.writer.close()
     

if __name__ == "__main__":
    main()