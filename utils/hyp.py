import numpy as np
import argparse
try:
    from torch_utils import select_device
except:
    from .torch_utils import select_device

import torch

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal', type=str)
    parser.add_argument('--net', dest='net',
                        help='resnet',
                        default='resnet', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="weights",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        default=True)                      
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether to perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--visdom', default=False, type=bool, 
                        help='Use visdom for loss visualization')
                        
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
                
    args = parser.parse_args()

    args.device, args.ng = select_device(is_head=True)
    if args.ng > 0:
        args.cuda = True
        if args.ng > 1:
            args.gpu_ids = [int(s) for s in range(args.ng)]
            args.mGPUs = True
            args.batch_size = args.ng * args.batch_size

    print('# parametes list:')
    for (key, value) in args.__dict__.items():
        print(key, '=', value)
    print('')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args

if __name__ == "__main__":
    parse_args()