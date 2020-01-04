import json
import os
import cv2
import numpy as np


def load_data(json_fname, flag=cv2.IMREAD_GRAYSCALE):
    with open(json_fname) as f:
        data = json.load(f)
    points = []
    imgs = []
    labels = []
    for d in data:
        labels.append(d['label'])
        points.append(d['points_1'])  # 2 img -> 1 label
        imgs.append(cv2.imread(d['img_1'], flag))
        points.append(d['points_2'])
        imgs.append(cv2.imread(d['img_2'], flag))
    return np.array(points, dtype=np.float), np.array(labels, dtype=np.int), imgs


def load_labels(json_fname):
    with open(json_fname) as f:
        data = json.load(f)
    return np.array([x['label'] for x in data], dtype=np.int)


def LFW_file_name(data_root, name, index, suffix='.jpg'):
    return os.path.join(data_root, 'lfw', name, '{:s}_{:04d}{:s}'.format(name, index, suffix))


def LFW_file_list(list, data_root=None, suffix='.jpg'):
    with open(list) as f:
        lines = f.readlines()
    output = []
    for i in range(int(lines[0].strip())):
        name, index = lines[i + 1].strip().split('\t')
        output.append(LFW_file_name(data_root, name, int(index), suffix=suffix))
    return output


def LFW_merge_file_lists(lists, data_root=None, suffix='.jpg'):
    output = set()
    for l in lists:
        output = output.union(set(LFW_file_list(l, data_root=data_root, suffix=suffix)))
    return sorted(list(output))
