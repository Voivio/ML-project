# To make N-fold dataset.
import os
import numpy as np
import argparse
import pickle
from tqdm import tqdm


def lfw_file_name(lfw_root, name, index, suffix='jpg'):
    return os.path.join(lfw_root, 'lfw', name, '{:s}_{:04d}.{:s}'.format(name, index, suffix))


def parse_line(line):
    line = line.strip().split('\t')
    if len(line) == 3:
        return line[0], int(line[1]), line[0], int(line[2]), 1
    else:
        return line[0], int(line[1]), line[2], int(line[3]), 0


def parse_one_fold(lines, lfw_root, feat_suffix):
    features = []
    labels = []
    skipped_labels = []
    for line in tqdm(lines):
        name1, idx1, name2, idx2, label = parse_line(line)
        feat1 = lfw_file_name(lfw_root, name1, idx1, suffix=feat_suffix)
        feat2 = lfw_file_name(lfw_root, name2, idx2, suffix=feat_suffix)
        if not os.path.exists(feat1) or not os.path.exists(feat2):
            skipped_labels.append(label)
        else:
            with open(feat1, 'rb') as f:
                feat1 = pickle.load(f)
            with open(feat2, 'rb') as f:
                feat2 = pickle.load(f)
            features.append([feat1, feat2])
            labels.append(label)
    labels = np.array(labels, dtype=np.int)
    skipped_labels = np.array(skipped_labels, dtype=np.int)
    print("Preserved %d data\t\t%d pos\t%d neg" % (labels.shape[0], (labels == 1).sum(), (labels == 0).sum()))
    print("Skipped %d data\t\t%d pos\t%d neg" % (skipped_labels.shape[0], (skipped_labels == 1).sum(),
                                                 (skipped_labels == 0).sum()))
    return np.array(features, dtype=np.float), labels, skipped_labels


def make_dataset(lfw_root, split_file, feat_suffix):
    with open(split_file) as f:
        lines = f.readlines()
    n_fold, n_each = lines[0].strip().split('\t')
    n_fold = int(n_fold)
    n_each = int(n_each)

    datas = []
    for i in range(n_fold):
        print("Parsing fold %d / %d" % (i, n_fold))
        datas.append(parse_one_fold(lines[n_each * i * 2 + 1:n_each * (i * 2 + 2) + 1], lfw_root, feat_suffix))
    print("\nIn total: skipped %d data, %d pos, %d neg\n" % (sum([x[-1].shape[0] for x in datas]),
                                                             sum([(x[-1] == 1).sum() for x in datas]),
                                                             sum([(x[-1] == 0).sum() for x in datas])))
    return datas


if __name__ == '__main__':
    # Make dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--suffix', required=True)
    args = parser.parse_args()

    datas = make_dataset(args.root, args.split_file, args.suffix)

    assert not os.path.exists(args.dump_dir)
    os.makedirs(args.dump_dir)
    for i, d in enumerate(datas):
        dump_fname = os.path.join(args.dump_dir, 'fold_{:d}.pkl'.format(i))
        with open(dump_fname, 'wb') as f:
            pickle.dump(d, f)
        print("Dump fold %d to %s" % (i, dump_fname))
