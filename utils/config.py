# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


cfg = edict()

"""
Dataset options
"""
# input size
cfg.input_size = 512

# background label
cfg.bkg_label = 0

# negative overlap
cfg.neg_overlap = 0.5

# Negative mining
cfg.neg_mining = True

# Negative position
cfg.neg_position = 3

# encode target
cfg.encode_targe = False


"""
Model options
"""
# Feature maps
cfg.feature_maps = [64, 32, 16, 16]

"""
Loss options
"""
# overlap threshold
cfg.overlap_thresh = 0.5
cfg.prior_for_matching = True

# Initial learning rate
cfg.lr = 0.001

# Momentum
cfg.momentum = 0.9

# Weight decay, for regularization
cfg.weight_decay = 0.0005

# Factor for reducing the learning rate
cfg.gamma = 0.1


"""
Anchor options
"""
# Anchor scales
cfg.anchor_scales = [32, 96, 160, 224, 288, 352, 416, 480]

# Anchor number of per layer
cfg.anchor_number = [2, 2, 2, 2]

# Anchor ratios
cfg.anchor_ratios = [0.5, 1, 2]

# clip anchor for keep anchor local in image
cfg.clip = True

# Variance
cfg.variance = [0.1, 0.2]