import json
import cv2
import numpy as np


def load_data(json_fname, flag=cv2.IMREAD_GRAYSCALE):
    with open(json_fname) as f:
        data = json.load(f)  #
    points = []
    imgs = []
    labels = []
    for d in data:
        points.append(d['points'])
        labels.append(d['label'])
        imgs.append(cv2.imread(d['img'], flag))
    return np.array(points, dtype=np.float), np.array(labels, dtype=np.int), imgs
