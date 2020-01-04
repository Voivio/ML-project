import argparse
import os
import cv2
import numpy as np

from classifier import get_classifier, CLASSIFIER_MAP
from compressor import get_compressor
from landmark_detector import get_detector
from facealigner import align  # TODO: we suppose we use default paras
from feature_utils import MB_LBP
from train import aggregate_data


def predict(img1, img2, detector, feature, agg, compressor, classifier, default_option):
    # img1
    pts1 = detector.detect(img1)
    if pts1 is None:
        return default_option
    img1, pts1 = align(img1, pts1)
    # img2
    pts2 = detector.detect(img2)
    if pts2 is None:
        return default_option
    img2, pts2 = align(img2, pts2)
    # feature
    feat1 = feature.extract(img1, pts1)
    feat2 = feature.extract(img2, pts2)
    # aggregate
    x = aggregate_data((np.array([[feat1, feat2]], dtype=float), None, None), agg)
    x = compressor.transform(x)
    return int(classifier.predict(x).squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('txt')
    parser.add_argument('--classifier', choices=list(CLASSIFIER_MAP.keys()))
    parser.add_argument('--load-compressor')
    parser.add_argument('--load-classifier')
    parser.add_argument('--load-default-option')
    parser.add_argument('--agg', default='abs-minus', choices=['minus-abs', 'mul_minus-abs'])
    args, _ = parser.parse_known_args()

    detector = get_detector()

    feature = MB_LBP(parser)

    compressor = get_compressor('pca', 2333)
    compressor.load(args.load_compressor)

    classifier = get_classifier(args.classifier)
    classifier.load(args.load_classifier)

    args = parser.parse_args()
    print(args)

    with open(args.load_default_option) as f:
        default_option = int(f.read().strip())

    output = {}
    for k in os.listdir(args.data):
        img1, img2 = os.listdir(os.path.join(args.data, k))
        img1 = cv2.imread(img1, os.path.join(args.data, k, img1))
        img2 = cv2.imread(img2, os.path.join(args.data, k, img2))
        output[k] = predict(img1, img2, detector, feature, args.agg, compressor, classifier, default_option)

    with open(args.txt, 'w') as f:
        for k, v in output.items():
            f.write('{}\t{:d}\n'.format(k, v))
