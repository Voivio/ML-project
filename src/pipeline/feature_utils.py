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


# from data_utils import load_data


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


def init_lbp_mapping():
    # TODO: Don't know why he does it
    lbp_mapping = [0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58,
                   13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58,
                   58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                   58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58,
                   58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
                   29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                   58, 58, 58, 58, 58, 58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                   58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58,
                   58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44,
                   58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54,
                   55, 56, 57]
    m = {}
    for i, x in enumerate(lbp_mapping):
        m["{:08b}".format(i)] = x
    return m


def lbp_hist(img, lbp_mapping_dict):
    h, w = img.shape
    data = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            local_pattern = [img[i - 1, j], img[i - 1, j - 1], img[i, j - 1], img[i + 1, j - 1], img[i + 1, j],
                             img[i + 1, j + 1], img[i, j + 1], img[i - 1, j + 1]]
            if lbp_mapping_dict is not None:
                raw_code = "".join([str(int(x > img[i, j])) for x in local_pattern])
                lbp_code = lbp_mapping_dict[raw_code]
            else:
                lbp_code = int("".join([str(int(x > img[i, j])) for x in local_pattern]), 2)
            data.append(lbp_code)
    return np.bincount(data, minlength=(len(lbp_mapping_dict) if lbp_mapping_dict is not None else (1 << 8)))


class HighDimensionalLBP:
    # Ref: https://github.com/bcsiriuschen/High-Dimensional-LBP/blob/master/src/LBPFeatureExtractor.cpp

    def __init__(self, parser: argparse.ArgumentParser):
        parser.add_argument('--scales', nargs='+', type=int, default=[300, 212, 150, 106, 75])
        parser.add_argument('--patch-size', type=int, default=10)
        parser.add_argument('--num-cell-x', type=int, default=4)
        parser.add_argument('--num-cell-y', type=int, default=4)
        parser.add_argument('--use-dict', action='store_true')
        args, _ = parser.parse_known_args()
        self.scales = args.scales
        self.patch_size = args.patch_size
        self.num_cell_x = args.num_cell_x
        self.num_cell_y = args.num_cell_y
        # self.uniform = uniform
        if args.use_dict:
            self.lbp_mapping_dict = init_lbp_mapping()
        else:
            self.lbp_mapping_dict = None

    @property
    def n_lbp(self):
        if self.lbp_mapping_dict is None:
            return (1 << 8)
        else:
            return len(self.lbp_mapping_dict)

    def extract(self, img, points, debug=False, detailed_debug=False):
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
        img_shape = np.array(img.shape, dtype=np.float)
        features = []
        for s in self.scales:
            new_points = points * s / img_shape[None, :]  # (N, 2)
            img_resized = cv2.resize(img_normalized, (s, s)) if s > 0 else img_normalized
            if debug:
                show_img_with_points(img_resized, new_points, 'rescaled to ({:d}, {:d})'.format(s, s))
            # LBP
            for x, y in new_points:
                for j in np.arange(self.num_cell_x):
                    for k in np.arange(self.num_cell_y):
                        # crop each patch
                        center_x = int(x - self.patch_size * self.num_cell_x / 2 + float(
                            self.num_cell_x % 2 == 0) * self.patch_size * 0.5 + self.patch_size * j)
                        center_y = int(y - self.patch_size * self.num_cell_y / 2 + float(
                            self.num_cell_y % 2 == 0) * self.patch_size * 0.5 + self.patch_size * k)
                        if detailed_debug:
                            show_img_with_points(img, {'crop center': [center_x, center_y], 'point': [x, y]},
                                                 'crop patch for cell {:d}, {:d}'.format(j, k), show=False)
                        patch = cv2.getRectSubPix(img_resized, (self.patch_size + 2, self.patch_size + 2),
                                                  (center_x, center_y))
                        if detailed_debug:
                            show_img_with_points(patch, None, 'cropped patch at cell {:d}, {:d}'.format(j, k),
                                                 show=False)
                        hist = lbp_hist(patch, self.lbp_mapping_dict)
                        if detailed_debug:
                            plt.figure()
                            plt.bar(x=range(self.n_lbp), height=hist)
                            plt.title("hist for cropped patch at cell {:d}, {:d}".format(j, k))
                            plt.show()
                        features += list(hist)
        return np.array(features, dtype=np.float)


class FasterHighDimensionalLBP:
    # Ref: https://github.com/bcsiriuschen/High-Dimensional-LBP/blob/master/src/LBPFeatureExtractor.cpp

    def __init__(self, parser: argparse.ArgumentParser):
        parser.add_argument('--scales', nargs='+', type=int, default=[1, 2, 3, 4, 5])
        parser.add_argument('--patch-size', type=int, default=10)
        parser.add_argument('--num-cell-x', type=int, default=4)
        parser.add_argument('--num-cell-y', type=int, default=4)
        args, _ = parser.parse_known_args()
        self.scales = args.scales
        self.patch_size = args.patch_size
        self.num_cell_x = args.num_cell_x
        self.num_cell_y = args.num_cell_y
        # self.uniform = uniform
        self.lbp_mapping_dict = init_lbp_mapping()

    def extract(self, img, points, debug=False, detailed_debug=False):
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
        # img_shape = np.array(img.shape, dtype=np.float)
        features = []
        for s in self.scales:
            new_points = points / float(s)
            img_resized = block_reduce(img, (s, s), func=np.max)  # TODO: or np.mean?
            if debug:
                show_img_with_points(img_resized, new_points, 'rescaled to ({:d}, {:d})'.format(s, s))
            # LBP
            for x, y in new_points:
                for j in np.arange(self.num_cell_x):
                    for k in np.arange(self.num_cell_y):
                        # crop each patch
                        center_x = int(x - self.patch_size * self.num_cell_x / 2 + float(
                            self.num_cell_x % 2 == 0) * self.patch_size * 0.5 + self.patch_size * j)
                        center_y = int(y - self.patch_size * self.num_cell_y / 2 + float(
                            self.num_cell_y % 2 == 0) * self.patch_size * 0.5 + self.patch_size * k)
                        if detailed_debug:
                            show_img_with_points(img, {'crop center': [center_x, center_y], 'point': [x, y]},
                                                 'crop patch for cell {:d}, {:d}'.format(j, k), show=False)
                        patch = cv2.getRectSubPix(img_resized, (self.patch_size + 2, self.patch_size + 2),
                                                  (center_x, center_y))
                        if detailed_debug:
                            show_img_with_points(patch, None, 'cropped patch at cell {:d}, {:d}'.format(j, k),
                                                 show=False)
                        hist = lbp_hist(patch, self.lbp_mapping_dict)
                        if detailed_debug:
                            plt.figure()
                            plt.bar(x=range(len(self.lbp_mapping_dict)), height=hist)
                            plt.title("hist for cropped patch at cell {:d}, {:d}".format(j, k))
                            plt.show()
                        features += list(hist)
        return np.array(features, dtype=np.float)


class MB_LBP:
    # Ref: https://blog.csdn.net/jxch____/article/details/80565601

    def __init__(self, parser: argparse.ArgumentParser):
        parser.add_argument('--scales', nargs='+', type=int, default=[1, 3, 7, 13, 25])
        args, _ = parser.parse_known_args()
        self.scales = args.scales

    def mb_lbp(self, img, x, y, bs):
        patch = cv2.getRectSubPix(img, (bs * 3, bs * 3), (x, y))
        patch = block_reduce(patch, (bs, bs), func=np.mean)
        return [float(patch[i, j] > patch[1, 1]) for i, j in
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]]

    def extract(self, img, points, debug=False, detailed_debug=False):
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


# def extract_and_dump(lbp: HighDimensionalLBP, data: str, output: str):
#     points, _, imgs = load_data(data, cv2.IMREAD_GRAYSCALE)
#     features = []
#     for img, pts in zip(imgs, points):
#         features.append(lbp.extract(img, pts))
#     assert not os.path.exists(output)
#     with open(output, 'wb') as f:
#         pickle.dump(features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--feat', default='high_dim', choices=['high_dim', 'faster_high_dim', 'mb_lbp'])
    parser.add_argument('--feat-suffix', required=True)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--detailed-debug', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    feat_cls = dict(
        high_dim=HighDimensionalLBP,
        faster_high_dim=FasterHighDimensionalLBP,
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
            features = feat_extractor.extract(img, pts, debug=args.debug, detailed_debug=args.detailed_debug)
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
