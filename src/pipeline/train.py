import os
import argparse
import pickle
import numpy as np
from sklearn import metrics  # TODO: need to implement them myself

# from data_utils import load_labels
from compressor import get_compressor
from classifier import get_classifier, CLASSIFIER_MAP

'''
Usage:
1. For original images: detect landmarks, transform images, and dump the data into the data_utils format (TODO)
2. For transformed images & landmarks: extract & dump LBP features using feature_utils.py
3. Train using this file
4. Infer (TODO)
'''


def aggregate_data(data, agg):
    x, y, z = data
    x1, x2 = x.transpose(1, 0, 2)
    if agg == 'minus-abs':
        return np.abs(x1 - x2), y, z
    else:
        raise ValueError("agg cannot be %s" % agg)


def evaluate(y, z, pred, default_option):
    print("\tw/o invalid data:")
    print("\t\tAcc score  = %.2f" % (metrics.accuracy_score(y, pred) * 100))
    print("\t\tF1 score   = %.2f" % (metrics.f1_score(y, pred) * 100))
    y = np.concatenate([y, z], axis=0)
    pred = np.concatenate([pred, np.repeat(default_option, z.shape)], axis=0)
    print("\twith invalid data:")
    acc = metrics.accuracy_score(y, pred) * 100
    f1 = metrics.f1_score(y, pred) * 100
    print("\t\tAcc score  = %.2f" % acc)
    print("\t\tF1 score   = %.2f" % f1)
    return acc, f1


def load_data(data):
    folds = []
    for i in range(len(os.listdir(data))):
        with open(os.path.join(data, 'fold_{:d}.pkl'.format(i)), 'rb') as f:
            folds.append(pickle.load(f))
    return folds


def get_fold(folds, test_fold: int):
    train = folds[:test_fold] + folds[test_fold + 1:]
    train = list(zip(*train))
    train = [np.concatenate(x, axis=0) for x in train]
    test = folds[test_fold]
    return train, test


def train_and_evaluate(train, test, args):
    train_x, train_y, train_z = train
    test_x, test_y, test_z = test

    compressor = get_compressor('pca', args.n_component)
    train_x = compressor.fit_transform(train_x)
    test_x = compressor.transform(test_x)
    compressor.save(args.save_compressor)

    classifier = get_classifier(args.classifier)
    classifier.fit(train_x, train_y)
    classifier.save(args.save_classifier)

    default_option = np.argmax([(train_z == 0).sum(), (train_z == 1).sum()])
    if args.save_default is not None:
        assert not os.path.exists(args.save_default)
        with open(args.save_default, 'w') as f:
            f.write(str(default_option))

    train_pred = classifier.predict(train_x)
    print("Evaluate on training set")
    evaluate(train_y, train_z, train_pred, default_option)
    test_pred = classifier.predict(test_x)
    print("Evaluate on testing set")
    return evaluate(test_y, test_z, test_pred, default_option)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # Possibly will configure: compressor, trainer
    parser.add_argument('--data')
    parser.add_argument('--test-fold', default=9, type=int)
    parser.add_argument('--classifier', required=True, choices=list(CLASSIFIER_MAP.keys()))
    parser.add_argument('--agg-feature', default='minus-abs', choices=['minus-abs', ])  # feature aggregation
    parser.add_argument('--n-component', type=int)  # compressor
    parser.add_argument('--save-compressor', default=None)  # save dir
    parser.add_argument('--save-classifier', default=None)
    parser.add_argument('--save-default', default=None)
    args = parser.parse_args()

    folds = load_data(args.data)
    if args.test_fold < 0:
        raise NotImplementedError
    else:
        assert args.test_fold < 10
        train, test = get_fold(folds, args.test_fold)
        train = aggregate_data(train, args.agg_feature)
        test = aggregate_data(test, args.agg_feature)
        train_and_evaluate(train, test, args)
