import os
import argparse
import pickle
import numpy as np

# from data_utils import load_labels
from pca import PCA
from logistic_regression import LogisticRegression
from model import Model

'''
Usage:
1. For original images: detect landmarks, transform images, and dump the data into the data_utils format (TODO)
2. For transformed images & landmarks: extract & dump LBP features using feature_utils.py
3. Train using this file
4. Infer
'''


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
    compressor = PCA(args.n_components)
    classifier = LogisticRegression(args.lr, args.iters, verbose=args.verbose)

    model = Model(args.agg_feature, compressor, classifier)

    model.train(train)

    print("Evaluate on training set")
    model.evaluate(train)
    print("Evaluate on testing set")
    model.evaluate(test)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # Possibly will configure: compressor, trainer
    parser.add_argument('--data')
    parser.add_argument('--test-fold', default=9, type=int)
    # PCA
    parser.add_argument('--n-components', default=256, type=int)
    # Logistic Regression
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--iters', default=1500, type=int)
    parser.add_argument('--verbose', action='store_true')
    # Model
    parser.add_argument('--agg-feature', default='minus-abs', choices=Model.AGG_CHOICES)
    parser.add_argument('--dump', default=None)
    args = parser.parse_args()

    folds = load_data(args.data)
    if args.test_fold < 0:
        raise NotImplementedError
    else:
        assert args.test_fold < 10
        train, test = get_fold(folds, args.test_fold)
        model = train_and_evaluate(train, test, args)
        if args.dump is not None:
            assert not os.path.exists(args.dump)
            with open(args.dump, 'wb') as f:
                pickle.dump(model, f)
