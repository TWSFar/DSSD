import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def create_vis_plot(vis, X_, Y_, title_, legend_):
    return vis.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(legend_))).cpu(),
        opts=dict(
            xlabel=X_,
            ylabel=Y_,
            title=title_,
            legend=legend_
        )
    )


def update_vis_plot(vis, item, loss, window, update_type):
    if item == 0:
        update_type = True
 
    vis.line(
        X = torch.ones((1, len(loss))).cpu() * item,
        Y = torch.Tensor(loss).unsqueeze(0).cpu(),
        win = window,
        update = update_type
    )


def model_info(model, report='summary'):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if report is 'full':
        print("%5s %40s %9s %12s %20s %10s %10s" % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %50g %9s %12g %20g %10.3g %10.3g' % 
                (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (len(list(model.parameters())), n_p, n_g))