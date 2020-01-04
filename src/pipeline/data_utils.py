# import json
import os
# import cv2
import numpy as np
import argparse
import pickle
from tqdm import tqdm


# def load_data(json_fname, flag=cv2.IMREAD_GRAYSCALE):
#     with open(json_fname) as f:
#         data = json.load(f)
#     points = []
#     imgs = []
#     labels = []
#     for d in data:
#         labels.append(d['label'])
#         points.append(d['points_1'])  # 2 img -> 1 label
#         imgs.append(cv2.imread(d['img_1'], flag))
#         points.append(d['points_2'])
#         imgs.append(cv2.imread(d['img_2'], flag))
#     return np.array(points, dtype=np.float), np.array(labels, dtype=np.int), imgs
#
#
# def load_labels(json_fname):
#     with open(json_fname) as f:
#         data = json.load(f)
#     return np.array([x['label'] for x in data], dtype=np.int)


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


def make_dataset_from_pkl(data_root, split_file, suffix, ignore_error=True):
    with open(split_file) as f:
        lines = f.readlines()
    n = int(lines[0].strip())

    def read(name, index):
        fname = LFW_file_name(data_root, name, int(index), suffix='.transformed' + suffix)
        if not os.path.exists(fname):
            if ignore_error:
                print("%s doesn't exist, skip it!" % fname)
                return None
            else:
                raise ValueError("%s not exist" % fname)
        else:
            with open(fname, 'rb') as f:
                return pickle.load(f)

    features = []
    labels = []
    cls = [0, 0]
    for i, line in tqdm(list(enumerate(lines[1:]))):
        if i < n:
            name1, idx1, idx2 = line.strip().split('\t')
            name2 = name1
            label = 1
        else:
            name1, idx1, name2, idx2 = line.strip().split('\t')
            label = 0
        x1 = read(name1, idx1)
        if x1 is None:
            continue
        x2 = read(name2, idx2)
        if x2 is None:
            continue
        features.append([x1, x2])
        labels.append(label)
        cls[label] += 1

    print("Data: total = %d, neg = %d, pos = %d" % (len(features), cls[0], cls[1]))
    return np.array(features, dtype=np.float), np.array(labels, dtype=np.int)


if __name__ == '__main__':
    # Make dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--dump-file', required=True)
    parser.add_argument('--ignore-error', action='store_true')
    parser.add_argument('--suffix', required=True)
    args = parser.parse_args()
    train_data, train_labels = make_dataset_from_pkl(args.data_root, args.split_file, ignore_error=args.ignore_error, suffix=args.suffix)
    assert not os.path.exists(args.dump_file)
    with open(args.dump_file, 'wb') as f:
        pickle.dump((train_data, train_labels), f)
    print("Dump to %s" % args.dump_file)
