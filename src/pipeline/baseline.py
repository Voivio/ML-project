import argparse
import cv2
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from data_utils import LFW_file_name


def dataset(data_root, split_file):
    with open(split_file) as f:
        lines = f.readlines()
    n = int(lines[0].strip())

    def read(name, index):
        img = LFW_file_name(data_root, name, int(index), suffix='.jpg')
        return cv2.equalizeHist(cv2.imread(img, 0)).reshape(-1)

    data = []
    labels = []
    for i, line in tqdm(list(enumerate(lines[1:]))):
        if i < n:
            name1, idx1, idx2 = line.strip().split('\t')
            name2 = name1
            label = 1
        else:
            name1, idx1, name2, idx2 = line.strip().split('\t')
            label = 0
        img1 = read(name1, idx1)
        img2 = read(name2, idx2)
        data.append([img1, img2])
        labels.append(label)

    return np.array(data, dtype=np.float), np.array(labels, dtype=np.int)


def evaluate(y_true, y_pred):
    print("\tacc = %.3f\n\tf1 = %.3f" % (sklearn.metrics.accuracy_score(y_true, y_pred),
                                         sklearn.metrics.f1_score(y_true, y_pred)))


def train_baseline(train_data, train_labels, test_data, test_labels, args):
    if args.n_pca > 0:
        pca = PCA(args.n_pca)
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)
    if args.model == 'svm':
        model = SVC()
    else:
        raise ValueError("model cannot be %s" % args.model)
    model.fit(train_data, train_labels)
    train_pred = model.predict(train_data)
    evaluate(train_labels, train_pred)
    test_pred = model.predict(test_data)
    evaluate(test_labels, test_pred)


def aggregate(x, agg):
    if agg == 'concat':
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError("agg cannot be %s" % agg)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('train')
    parser.add_argument('test')
    parser.add_argument('--model', default='svm', choices=['svm'])
    parser.add_argument('--n-pca', default=-1, type=int)
    parser.add_argument('--agg', default='concat', choices=['concat'])
    args = parser.parse_args()

    train_data, train_labels = dataset(args.data_root, args.train)
    test_data, test_labels = dataset(args.data_root, args.test)
    train_data = aggregate(train_data, args.agg)
    test_data = aggregate(test_data, args.agg)

    train_baseline(train_data, train_labels, test_data, test_labels, args)
