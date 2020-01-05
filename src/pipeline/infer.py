import argparse
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from landmark_detector import get_detector
from facealigner import align, disp_landmarks  # TODO: we suppose we use default paras
from feature_utils import MB_LBP


def predict(img1, img2, detector, feature, model, debug=False):
    # img1
    pts1 = detector.detect(img1)
    if pts1 is None:
        return model.default_option
    if debug:
        disp_landmarks(img1, pts1, title="orig img1")
    img1, pts1 = align(img1, pts1)
    if debug:
        disp_landmarks(img1, pts1, title="aligned img1")
    # img2
    pts2 = detector.detect(img2)
    if pts2 is None:
        return model.default_option
    if debug:
        disp_landmarks(img2, pts2, title="orig img2")
    img2, pts2 = align(img2, pts2)
    if debug:
        disp_landmarks(img2, pts2, title="aligned img2")
        plt.show()
    # feature
    feat1 = feature.extract(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), pts1, debug=debug)
    feat2 = feature.extract(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), pts2, debug=debug)
    # aggregate
    feat = np.array([[feat1, feat2]], dtype=float)
    return int(model.infer(feat))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('txt')
    parser.add_argument('--model')
    parser.add_argument('--debug', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    detector = get_detector()
    feature = MB_LBP(parser)
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    args = parser.parse_args()
    print(args)

    output = {}
    for k in tqdm(sorted(list(os.listdir(args.data)))):
        img1, img2 = os.listdir(os.path.join(args.data, k))
        img1 = cv2.imread(os.path.join(args.data, k, img1))
        img2 = cv2.imread(os.path.join(args.data, k, img2))
        output[k] = predict(img1, img2, detector, feature, model, debug=args.debug)

    with open(args.txt, 'w') as f:
        for k, v in output.items():
            print(k)
            f.write("{:d}\n".format(v))
            # f.write('{}\t{:d}\n'.format(k, v))
