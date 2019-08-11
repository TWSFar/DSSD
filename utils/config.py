from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

"""
Dataset options
"""
# Image size
__C.IMAGE_SIZE = 512

# background label
__C.BKG_LABEL = 0

# negative overlap
__C.NEG_OVERLAP = 0.5

# Negative ming
__C.NEG_MING = True

# Negative position
__C.NEG_POSITION = 3

# encode target
__C.ENCODE_TARGE = False


"""
Model options
"""
# Feature maps
__C.FEATURE_MAPS = [64, 32, 16, 16]


"""
Loss options 
"""
# overlap threshold
__C.OVERLAP_THRESH = 0.5

#
__C.PRIOR_FOR_MATCHING = True


"""
Training options
"""
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1


"""
Testing options
"""
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Maximal number of gt rois in an image during Training
__C.MAX_NUM_GT_BOXES = 20


"""
Anchor options
"""
# Anchor scales 
__C.ANCHOR_SCALES = [32, 96, 160, 224, 288, 352, 416, 480]

# Anchor number of per layer
__C.ANCHOR_NUMBER = [2, 2, 2, 2]

# Anchor ratios
__C.ANCHOR_RATIOS = [0.5, 1, 2]

# clip anchor for keep anchor local in image
__C.CLIP = True

# Variance
__C.VARIANCE = [0.1, 0.2]


