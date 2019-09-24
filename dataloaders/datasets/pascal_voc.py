import os
import PIL
import pickle
import os.path as osp
import numpy as np
import scipy.sparse
import xml.etree.ElementTree as ET
from PIL import Image
try:
    from mypath import Path
except:
    print('test...')
    import sys
    sys.path.extend(['G:\\CV\\Reading\\DSSD',])
    from mypath import Path
    

class pascal_voc(object):

    num_classes = 21

    def __init__(self,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 mode='train'):
        super().__init__()
        self.mode = mode
        self._base_dir = base_dir
        self._img_dir = osp.join(self._base_dir, 'JPEGImages')
        self._cat_dir = osp.join(self._base_dir, 'Annotations')
        self.cache_path = self.cre_cache_path()
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))

        # PASCAL image index
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self.im_ids = self._load_image_set_index()
        self.num_images = len(self.im_ids)
        
        # label
        self.roidb = self.gt_roidb()
        self.prepare_roidb()
        if self.mode == 'train':
            self.filter_roidb()

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

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
        filename = os.path.join(self._base_dir, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float64)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        for split in self.split:
            image_set_file = os.path.join(self._base_dir, 'ImageSets', 'Main',
                                          split + '.txt')
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                for line in f.readlines():
                    image_index.append(line.strip())
        return image_index

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
            sizes = [PIL.Image.open(self.image_path_at(index)).size
                     for index in self.im_ids]
            with open(cache_file, 'wb') as f:
                pickle.dump(sizes, f)
            print('Done!!')

        for i, index in enumerate(self.im_ids):
            self.roidb[i]['img_id'] = self.image_id_at(i)
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
        return osp.join(self._img_dir, index + '.jpg')

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

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

    imdb = pascal_voc()
    print(imdb.roidb[0]['image'])
    pass
