import os
import PIL
import pickle
import os.path as osp
import numpy as np
import scipy.sparse
import xml.etree.ElementTree as ET
from PIL import Image
from pycocotools.coco import COCO
import sys
try:
    from mypath import Path
except:
    print('test...')
    sys.path.extend(['/home/twsf/work/DSSD/',])
    from mypath import Path


ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

def coco_class_weights():  # frequency of each class in coco train2014
    n = [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380]
    weights = 1 / torch.Tensor(n)
    weights /= weights.sum()
    # with open('data/coco.names', 'r') as f:
    #     for k, v in zip(f.read().splitlines(), n):
    #         print('%20s: %g' % (k, v))
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

class coco(object):

    num_classes = 81

    def __init__(self,
                 base_dir=Path.db_root_dir('coco'),
                 split='val2017',
                 mode='train'):
        super().__init__()
        self.mode = mode
        self._base_dir = base_dir
        self._base_dir = '/home/twsf/work/DSSD/data/COCO'
        self._img_dir = osp.join(self._base_dir, split)
        self._cat_dir = osp.join(self._base_dir, 'annotations')
        self.cache_path = self.cre_cache_path()
        self._classes = COCO_CLASSES
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._ind_to_class = dict(zip(range(self.num_classes), self._classes))
        self.cat_id = coco80_to_coco91_class()
        self.catid_to_ind = dict(zip(self.cat_id, range(self.num_classes)))
        self.coco = COCO(osp.join(self._cat_dir, INSTANCES_SET.format(split)))

        self.im_ids = list(self.coco.imgToAnns.keys())
        self.num_images = len(self.im_ids)

        # label
        self.roidb = self.gt_roidb()
        self.prepare_roidb()
        if self.mode == 'train':
            self.filter_roidb()

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.mode + 'gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.mode, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.im_ids]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        target = self.coco.imgToAnns[index]
        ann_ids = self.coco.getAnnIds(imgIds=index)
        num_objs = len(target)

        boxes = np.zeros((num_objs, 4), dtype=np.float64)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(target):
            bbox = obj["bbox"]
            # Make pixel indexes 0-based
            x1 = float(bbox[0] - bbox[2] / 2.0)
            y1 = float(bbox[1] - bbox[3] / 2.0)
            x2 = float(bbox[0] + bbox[2] / 2.0)
            y2 = float(bbox[1] + bbox[3] / 2.0)

            cls = self.catid_to_ind[obj["category_id"]]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'seg_areas': seg_areas}

    def prepare_roidb(self):
        """
        Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        cache_file = osp.join(self.cache_path, self.mode + '_sizes.pkl')
        if os.path.exists(cache_file):
            print('Image sizes loaded from %s' % cache_file)
            with open(cache_file, 'rb') as f:
                sizes = pickle.load(f)
        else:
            print('Extracting image sizes... (It may take long time)')
            sizes = [[self.coco.imgs[index]["width"],
                     self.coco.imgs[index]["height"]]
                     for index in self.im_ids]
            with open(cache_file, 'wb') as f:
                pickle.dump(sizes, f)
            print('Done!!')

        for i, index in enumerate(self.im_ids):
            self.roidb[i]['img_id'] = index
            self.roidb[i]['image'] = self.image_path_at(index)
            self.roidb[i]['width'] = sizes[i][0]
            self.roidb[i]['height'] = sizes[i][1]
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = self.roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            self.roidb[i]['max_classes'] = max_classes
            self.roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)

    def image_path_at(self, index):
        """
        return image path of index
        """
        return osp.join(self._img_dir, self.coco.imgs[index]["file_name"])

    def filter_roidb(self):
        # filter the image without bounding box.
        print('before filtering, there are %d images...' % (len(self.roidb)))
        i = 0
        while i < len(self.roidb):
            if len(self.roidb[i]['boxes']) == 0:
                del self.roidb[i]
                i -= 1
            i += 1
        print('after filtering, there are %d images...\n' % (len(self.roidb)))

    def cre_cache_path(self):
        cache_path = osp.join(self._base_dir, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path


if __name__ == "__main__":

    imdb = coco()
    print(imdb.roidb[0]['image'])
    pass
