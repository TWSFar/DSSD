from dataloaders.Detection_Dataset import Detection_Dataset
from model.DSSD import DSSD
from model.multibox_loss import MultiBoxLoss
from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

import torch

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = cfg
        # Define Saver

        # Define Visdom

        # Define Dataloader
        self.train_loader = Detection_Dataset(args, cfg, 'train', 'train')
        self.val_loader = Detection_Dataset(args, cfg, 'test', 'val')

        # Define Network
        # initilize the network here.
        if args.net == 'resnet':
            model = DSSD(cfg=cfg,
                         backbone=args.net,
                         output_stride=32,
                         num_classes=self.train_loader.num_classes,
                         img_size=self.train_loader.input_size,
                         pretrained=True, 
                         mode='train')
        else:
            NotImplementedError
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        opetimizer = torch.optim.SGD(train_params, 
                                     momentum=cfg.TRAIN.LEARNING_RATE, 
                                     weight_decay=cfg.TRAIN.WEIGHT_DECAY, 
                                     nesterov=False)
        # Define Criterion
        # Whether to use class balanced weights

        # Define Evaluater
        
        # Define lr scherduler
         
        # Resuming Checkpoint

        # Using cuda

        # Clear start epoch if fine-tuning


def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval -  1):
            trainer.validation(epoch)
    
    trainer.writer.close()
     

if __name__ == "__main__":
    main()