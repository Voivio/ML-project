import argparse
import os
import numpy as np
import cv2
import pickle
import json
import time
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from skimage.measure import block_reduce


def show_img_with_points(img, points, title=None, show=True):
    plt.figure()
    plt.imshow(img, cmap='gray', vmax=255, vmin=0)

    def disp_points(pts, **kwargs):
        pts = np.array(pts, dtype=np.float)
        if len(pts.shape) == 1:
            pts = pts.reshape(1, -1)
        plt.scatter(pts[:, 0], pts[:, 1], **kwargs)

    if points is not None:
        if isinstance(points, dict):
            legends = []
            for name, pts in points.items():
                legends.append(name)
                disp_points(pts, marker='x')
            plt.legend(legends)
        else:
            disp_points(points, color='red', marker='x')
    if title:
        plt.title(title)
    if show:
        plt.show()


class MB_LBP:
    # Ref: https://blog.csdn.net/jxch____/article/details/80565601

    def __init__(self, parser: argparse.ArgumentParser):
        parser.add_argument('--scales', nargs='+', type=int, default=[3, 9, 17, 25, 49])
        args, _ = parser.parse_known_args()
        self.scales = args.scales

    def mb_lbp(self, img, x, y, bs):
        patch = cv2.getRectSubPix(img, (bs * 3, bs * 3), (x, y))
        patch = block_reduce(patch, (bs, bs), func=np.mean)
        return [float(patch[i, j] > patch[1, 1]) for i, j in
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]]

    def extract(self, img, points, debug=False):
        # img: (H, W)
        # points: (N, 2)
        assert len(img.shape) == 2  # (H, W)
        if debug:
            show_img_with_points(img, points, 'orig')
        # normalize
        img_normalized = cv2.equalizeHist(img)
        if debug:
            show_img_with_points(img_normalized, points, 'normalized')
        # extract LBP
        features = []
        for s in self.scales:
            for x, y in points:
                features += self.mb_lbp(img_normalized, x, y, s)
        return np.array(features, dtype=np.float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--feat', default='high_dim', choices=['high_dim', 'faster_high_dim', 'mb_lbp'])
    parser.add_argument('--feat-suffix', required=True)
    parser.add_argument('--debug', default=False, action='store_true')
    # parser.add_argument('--detailed-debug', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    feat_cls = dict(
        mb_lbp=MB_LBP
    )[args.feat]
    feat_extractor = feat_cls(parser)

    if os.path.isdir(args.data):
        jpg_list = sorted(list(glob.glob('{}/lfw/*/*_????.transformed.jpg'.format(args.data))))
        # further parse args
        args = parser.parse_args()
        print(args)
        # extract & dump
        for jpg in tqdm(jpg_list):
            img = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
            with open(jpg.replace('.jpg', '.json')) as f:
                pts = np.array(json.load(f), dtype=np.float)
            features = feat_extractor.extract(img, pts, debug=args.debug)  # , detailed_debug=args.detailed_debug)
            assert not os.path.exists(jpg.replace('.jpg', args.feat_suffix)), jpg.replace('.jpg', args.feat_suffix)
            with open(jpg.replace('.jpg', args.feat_suffix), 'wb') as f:
                pickle.dump(features, f)
    else:
        # further parse args
        parser.add_argument('--points-json', default=None)
        parser.add_argument('--n-points', default=27, type=int)
        args = parser.parse_args()
        print(args)
        # debug
        img = cv2.imread(args.data, cv2.IMREAD_GRAYSCALE)
        if args.points_json is not None:
            with open(args.points_json) as f:
                pts = json.load(f)
            if isinstance(pts, dict):
                pts = pts['points']
            pts = np.array(pts, dtype=np.float)
        else:
            pts = (np.random.rand(args.n_points, 2) * 0.8 + 0.05) * np.array(img.shape, dtype=np.float)[None, :]
        start = time.time()
        features = feat_extractor.extract(img, pts, debug=args.debug, detailed_debug=args.detailed_debug)
        print("Time elapsed %f" % (time.time() - start))
        print("Feature shape", features.shape)
