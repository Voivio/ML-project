import argparse
import os
import numpy as np
import cv2
import pickle
import time
import matplotlib.pyplot as plt

from data_utils import load_data


def show_img_with_points(img, points, title=None, show=True):
    plt.figure()
    plt.imshow(img, cmap='gray', vmax=255, vmin=0)

    def disp_points(pts, **kwargs):
        pts = np.array(pts, dtype=np.float)
        if len(pts.shape) == 1:
            pts = pts.reshape(1, -1)
        plt.scatter(pts[:, 1], pts[:, 0], **kwargs)

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


class HighDimensionalLBP:
    # Ref: https://github.com/bcsiriuschen/High-Dimensional-LBP/blob/master/src/LBPFeatureExtractor.cpp

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

    # TODO: Don't know why he does it

    def __init__(self, scales: np.ndarray, patch_size: int, num_cell_x: int, num_cell_y: int):
        self.scales = scales
        self.patch_size = patch_size
        self.num_cell_x = num_cell_x
        self.num_cell_y = num_cell_y
        # self.uniform = uniform
        self.lbp_mapping_dict = self.init_lbp_mapping()

    def init_lbp_mapping(self):
        m = {}
        for i, x in enumerate(self.lbp_mapping):
            m["{:08b}".format(i)] = x
        return m

    def lbp_hist(self, img):
        h, w = img.shape
        hist = np.zeros((len(self.lbp_mapping),), dtype=np.int)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                local_pattern = [img[i - 1, j], img[i - 1, j - 1], img[i, j - 1], img[i + 1, j - 1], img[i + 1, j],
                                 img[i + 1, j + 1], img[i, j + 1], img[i - 1, j + 1]]
                raw_code = "".join([str(int(x > img[i, j])) for x in local_pattern])
                lbp_code = self.lbp_mapping_dict[raw_code]
                hist[lbp_code] += 1
        return hist

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
            img_resized = cv2.resize(img_normalized, (s, s))
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
                        hist = self.lbp_hist(patch)
                        if detailed_debug:
                            plt.figure()
                            plt.bar(x=range(len(self.lbp_mapping)), height=hist)
                            plt.title("hist for cropped patch at cell {:d}, {:d}".format(j, k))
                            plt.show()
                        features += list(hist)
        return np.array(features, dtype=np.float)


def extract_and_dump(lbp: HighDimensionalLBP, data: str, output: str):
    points, _, imgs = load_data(data, cv2.IMREAD_GRAYSCALE)
    features = []
    for img, pts in zip(imgs, points):
        features.append(lbp.extract(img, pts))
    assert not os.path.exists(output)
    with open(output, 'wb') as f:
        pickle.dump(features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--scales', nargs='+', type=int, default=[300, 212, 150, 106, 75])
    parser.add_argument('--patch-size', type=int, default=10)
    parser.add_argument('--num-cell-x', type=int, default=4)
    parser.add_argument('--num-cell-y', type=int, default=4)
    args, unknown = parser.parse_known_args()

    lbp = HighDimensionalLBP(args.scales, args.patch_size, args.num_cell_x, args.num_cell_y)

    if args.data.endswith('.json'):
        # further parse args
        parser.add_argument('--output', default=None)
        args = parser.parse_args()
        print(args)
        # extract & dump
        extract_and_dump(lbp, args.data, args.output)
    else:
        # further parse args
        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--detailed-debug', default=False, action='store_true')
        parser.add_argument('--n-points', default=27, type=int)
        args = parser.parse_args()
        print(args)
        # debug
        img = cv2.imread(args.data, cv2.IMREAD_GRAYSCALE)
        pts = (np.random.rand(args.n_points, 2) * 0.8 + 0.05) * np.array(img.shape, dtype=np.float)[None, :]
        start = time.time()
        features = lbp.extract(img, pts, debug=args.debug, detailed_debug=args.detailed_debug)
        print("Time elapsed %f" % (time.time() - start))
        print("Feature shape", features.shape)
