from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_imgs(img_path, bbox=None,
             transform=None, normalize=None):
    assert os.path.isfile(img_path)
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.6)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    img = normalize(img)
    return img


class BirdDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 image_size=256, is_flip=True):
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trs = [transforms.Resize([image_size,image_size], interpolation=Image.ANTIALIAS)]
        if is_flip:
            trs.append(transforms.RandomHorizontalFlip())
        self.transform =  transforms.Compose(trs)

        self.data = []
        self.data_dir = data_dir
        self.split = split
        self.bbox = self.load_bbox()

        self.filenames = self.load_filenames(data_dir, 'train')
        self.test_filenames = self.load_filenames(data_dir, 'test')
        self.class_id = self.load_class_id(data_dir, 'train')
        self.test_class_id = self.load_class_id(data_dir, 'test')
        self.m = self.map_class_id()
        self.attr = self.load_attr()
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_class_id(self, data_dir, split):
        with open(os.path.join(data_dir, split, 'class_info.pickle'), 'rb') as f:
            class_id = pickle.load(f, encoding='iso-8859-1')
        return class_id

    def map_class_id(self):
        s = list(set(self.class_id))
        print('num of class: ', len(s))
        m = {}
        ind = 0
        for i in s:
            m[i] = ind
            ind += 1
        return m

    def load_attr(self):
        data_dir = self.data_dir
        attr_path = os.path.join(data_dir, 'CUB_200_2011/attributes/image_attribute_labels.txt')
        df_attrs = pd.read_csv(attr_path,
                                        delim_whitespace=True,
                                        header=None)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_attr = {img_file[:-4]: 0 for img_file in filenames}
        numImgs = len(filenames)
        class_ids = {img_file[:3]: 0 for img_file in filenames}
        class_attrs = {img_file[:3]: 0 for img_file in filenames}
        for i in range(0, numImgs):
            attr = df_attrs.iloc[i * 312 : (i + 1) * 312][2].astype(int).tolist()
            key = filenames[i][:-4]
            filename_attr[key] = attr
        #
        print('filename_attr: ', len(filename_attr))
        return filename_attr

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames
        
    def __getitem__(self, index):
        if self.split == 'train':
            key = self.filenames[index]
            cls_id = self.class_id[index]
            cls_id = self.m[cls_id]
            attr = torch.FloatTensor(self.attr[key])
            attr = torch.clamp(attr, 0.0, 1.0)
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            img = get_imgs(img_name, bbox, self.transform, normalize=self.norm)

            # get unseen_attr
            index_u = index if index < len(self.test_filenames) else index % len(self.test_filenames)
            key_u = self.test_filenames[index_u]
            attr_u = torch.FloatTensor(self.attr[key_u])
            attr_u = torch.clamp(attr_u, 0.0, 1.0)

            return img, cls_id, attr, attr_u
        elif self.split == 'test':
            key = self.filenames[index]
            cls_id = self.class_id[index]
            attr = torch.FloatTensor(self.attr[key])
            attr = torch.clamp(attr, 0.0, 1.0)
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            img = get_imgs(img_name, bbox, self.transform, normalize=self.norm)

            # For unseen class
            index_u = index if index < len(self.test_filenames) else index % len(self.test_filenames)
            key_u = self.test_filenames[index_u]
            cls_id_u = self.test_class_id[index_u]
            attr_u = torch.FloatTensor(self.attr[key_u])
            attr_u = torch.clamp(attr_u, 0.0, 1.0)
            bbox_u = self.bbox[key_u]
            img_name_u = '%s/images/%s.jpg' % (data_dir, key_u)
            img_u = get_imgs(img_name_u, bbox_u, self.transform, normalize=self.norm)

            return img, cls_id, attr, img_u, cls_id_u, attr_u

    def __len__(self):
        return len(self.filenames)
