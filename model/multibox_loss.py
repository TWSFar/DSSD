import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from utils.box_utils import match, log_sum_exp
except:
    import sys
    sys.path.append('G:\\CV\\Reading\\DSSD')
    from utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
    MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    """

    def __init__(self, args, cfg, num_classes, weight=None):
        super(MultiBoxLoss, self).__init__()

        self.device = args.device
        self.num_classes = num_classes
        self.threshold = cfg.overlap_thresh
        self.background_label = cfg.bkg_label
        self.encode_targe = cfg.encode_targe
        self.use_prior_for_matching = cfg.prior_for_matching
        self.do_neg_mining = cfg.neg_mining
        self.negpos_ratio = cfg.neg_position
        self.neg_overlap = cfg.neg_overlap
        self.variance = cfg.variance
        self.weight = weight

    def forward(self, preds, targets):
        loc_data, conf_data, priors = preds
        bs = loc_data.size(0)  # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(bs, num_priors, 4)
        conf_t = torch.LongTensor(bs, num_priors)
        for idx in range(bs):
            truths = targets[idx][:, :-1].data
            label = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, label, defaults, 
                  self.variance, loc_t, conf_t, idx)

        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch, num_priors, 4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # if diff both classes is small, the loss_c will be large
        batch_conf = conf_data.view(-1, num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(bs, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)  # idx is more small,  loss is more large
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, weight=self.weight, reduction='sum')

        # sum of losses: L(x, c, l, g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
