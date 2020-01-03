import json
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
