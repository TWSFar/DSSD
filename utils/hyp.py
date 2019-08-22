import numpy as np
import argparse
import torch


def select_device(force_cpu=False, is_head=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    ng = 0
    if not cuda:
        print('Using CPU\n')
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        if is_head:
            for i in range(ng):
                print('Using CUDA device{} _CudaDeviceProperties(name={}, total_memory={}MB'.\
                        format(i, x[i].name, round(x[i].total_memory/c)))
            print('')
    return device, ng


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DSSD Training")
    parser.add_argument('--net', type=str, default='resnet',
                    choices=['resnet',],
                    help='backbone name (default: resnet)')
    parser.add_argument('--output-stride', type=int, default=32,
                    help='network output stride (default: 32)')
    parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco'],
                    help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                    help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                    help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                    help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
    parser.add_argument('--visdom', default=False, type=bool,
                    help='Use visdom for loss visualization')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=2,
                    metavar='N', help='input batch size for \
                            testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true', default=False, 
                    help='ables CUDA training')
    parser.add_argument('--use-multi-gpu', default=True, type=bool,
                    help='use multiple gpu')
    parser.add_argument('--ng', type=int, default=0, help='number of gpu')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

    # checking point and save model path
    parser.add_argument('--is-save', type=bool, default=True,
                    help='save chekpoint (default: True)')
    parser.add_argument('--work-dirs', type=str, default='work_dirs',
                    help='save model path')
    parser.add_argument('--resume', type=str, default=None,
                    help='resuming path')
    parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')

    # evaluation option
    parser.add_argument('--validate', type=int, default=1,
                    help='evaluuation interval (default: 1)')
    parser.add_argument('--eval_from', type=str, default=None,
                    help='evaluate model file path')
    parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')
    parser.add_argument('--eval-batch-size', type=int, default=2,
                    metavar='N', help='input batch size for \
                            evaluator (default: auto)')
    args = parser.parse_args()

    args.device, args.ng = select_device(is_head=True) 
    # args.ng = 1
    if args.ng > 0:
        args.cuda = True
        args.gpu_ids = [int(s) for s in range(args.ng)]

    if args.sync_bn is None:
        args.sync_bn = True if args.ng > 1 else False

    if args.epochs is None:
        epochs = {
            'coco': 30,
            'pascal': 50,
    }
        args.epochs = epochs[args.dataset.lower()]
    if args.batch_size is None:
        args.batch_size = max(4 * args.ng, 1)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = 'dssd-' + str(args.net)

    print('# parametes list:')
    for (key, value) in args.__dict__.items():
        print(key, '=', value)
    print('')

    torch.manual_seed(args.seed)

    return args


if __name__ == "__main__":
    parse_args()