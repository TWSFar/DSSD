import os
import os.path as osp
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):
    
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(osp.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(osp.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = osp.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if osp.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, osp.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        
    def save_experiment_config(self):
        logfile = osp.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        log_file.write('hyp:\n')
        for key, val in self.args.__dict__.items():
            log_file.write(key + ':' + str(val) + '\n')
        
        log_file.write('\nconfig:\n')
        for key, val in self.cfg.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()