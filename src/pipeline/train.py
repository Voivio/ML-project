import argparse
import pickle
import numpy as np
from sklearn import metrics  # TODO: need to implement them myself

from data_utils import load_labels
from compress_feature import PCA
from classifiers import LogisticRegressionClassifier

'''
Usage:
1. For original images: detect landmarks, transform images, and dump the data into the data_utils format (TODO)
2. For transformed images & landmarks: extract & dump LBP features using feature_utils.py
3. Train using this file
4. Infer (TODO)
'''


def load_pair_feature(fname):
    with open(fname, 'rb') as f:
        features = pickle.load(f)
    return features.reshape(2, features.shape[0] / 2, features.shape[1])


def aggregate_data(x, y, agg):
    x1, x2 = x
    if agg == 'minus-abs':
        return np.abs(x1 - x2), y
    else:
        raise ValueError("agg cannot be %s" % agg)


def evaluate(classifier, x, y):
    y_pred = classifier.predict(x)
    print("\tAcc score  = %.3f" % metrics.accuracy_score(y, y_pred))
    print("\tF1 score   = %.3f" % metrics.f1_score(y, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # Possibly will configure: compressor, trainer
    parser.add_argument('--train-features')  # data
    parser.add_argument('--train-data')
    parser.add_argument('--test-features')
    parser.add_argument('--test-data')
    parser.add_argument('--agg-feature', default='minus-abs', choices=['minus-abs', ])  # feature aggregation
    parser.add_argument('--n-compressed-dim', type=int)  # compressor
    parser.add_argument('--save-compressor')  # save dir
    parser.add_argument('--save-classifier')
    args = parser.parse_args()

    train_x, train_y = aggregate_data(load_pair_feature(args.train_features), load_labels(args.train_data), args.agg)
    test_x, test_y = aggregate_data(load_pair_feature(args.test_features), load_labels(args.test_data), args.agg)

    compressor = PCA(args.n_compressed_dim)
    train_x = compressor.fit_transform(train_x)
    test_x = compressor.transform(test_x)
    compressor.save(args.save_compressor)

    classifier = LogisticRegressionClassifier()
    classifier.fit(train_x, train_y)
    classifier.save(args.save_classifier)

    print("Evaluate on training set")
    evaluate(classifier, train_x, train_y)
    evaluate(classifier, test_x, test_y)
