import json
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


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


def make_dataset(data_root, split_file, ignore_error=True):
    with open(split_file) as f:
        lines = f.readlines()
    n = int(lines[0].strip())

    def read(name, index):
        img = LFW_file_name(data_root, name, int(index), suffix='.transformed.jpg')
        pts = LFW_file_name(data_root, name, int(index), suffix='.transformed.json')
        if not os.path.exists(pts):
            if ignore_error:
                print("%s doesn't exist, skip it!" % pts)
                return img, None
            else:
                raise ValueError("%s not exist" % pts)
        else:
            with open(pts) as f:
                pts = json.load(f)
            return img, pts

    data = []
    cls = [0, 0]
    for i, line in tqdm(list(enumerate(lines[1:]))):
        if i < n:
            name1, idx1, idx2 = line.strip().split('\t')
            name2 = name1
            label = 1
        else:
            name1, idx1, name2, idx2 = line.strip().split('\t')
            label = 0
        img1, pts1 = read(name1, idx1)
        if pts1 is None:
            continue
        img2, pts2 = read(name2, idx2)
        if pts2 is None:
            continue
        data.append(dict(label=label, points_1=pts1, img_1=img1, points_2=pts2, img_2=img2))
        cls[label] += 1

    print("Data: total = %d, neg = %d, pos = %d" % (len(data), cls[0], cls[1]))
    return data


if __name__ == '__main__':
    # Make dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--dump-file', required=True)
    parser.add_argument('--ignore-error', action='store_true')
    args = parser.parse_args()
    data = make_dataset(args.data_root, args.split_file, ignore_error=args.ignore_error)
    assert not os.path.exists(args.dump_file)
    with open(args.dump_file, 'w') as f:
        json.dump(data, f)
    print("Dump to %s" % args.dump_file)
