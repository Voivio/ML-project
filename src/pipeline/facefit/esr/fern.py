r"""
Implementation of a random fern, used for regression.
The fern is learnt a mapping from vectors of pixels to corresponding
vectorial delta shapes.
"""

import numpy as np
import cv2

from ..util import rand_unit_vector


class FernBuilder:
    def __init__(self, n_features, beta):
        self.n_features = n_features
        self.beta = beta

    # Returns the feature (pair of pixels) that has the highest correlation with the targets.
    @staticmethod
    def _highest_correlated_feature(dir, targets, cov_pp, pixel_values, pixel_averages, var_pp_sum):
        n_pixels = len(cov_pp)
        # Project each target onto random direction.
        lengths = targets.dot(dir)
        cov_l_p = pixel_values.dot(lengths) / len(targets) - np.average(lengths) * pixel_averages
        correlation = (cov_l_p[:, None] - cov_l_p) / np.sqrt(np.std(lengths) * (var_pp_sum - 2 * cov_pp))

        res = np.nanargmax(correlation)
        return res / n_pixels, res % n_pixels

    # Calculates the average offset within each bin of the fern.
    @staticmethod
    def _calc_bin_averages(targets, bin_ids, n_features, n_landmarks, beta):
        bins = np.zeros((2 ** n_features, 2 * n_landmarks))
        bins_size = np.zeros((2 ** n_features,))
        for i in range(len(bin_ids)):
            bins[bin_ids[i]] += targets[i]
            bins_size[bin_ids[i]] += 1
        denoms = (bins_size + beta)
        # Replace all 0 denominators with 1 to avoid division by 0.
        denoms[denoms == 0] = 1
        return bins / denoms[:, None]

    # Trains the fern on the given training data.
    def build(self, pixel_samples, targets, data):
        n_landmarks = targets.shape[1] / 2
        cov_pp, pixel_values, pixel_averages, pixel_var_sum = data
        feature_indices = np.zeros((self.n_features, 2), dtype=int, order='c')

        for f in range(self.n_features):
            dir = rand_unit_vector(2 * n_landmarks)
            feature_indices[f] = FernBuilder._highest_correlated_feature(dir, targets, cov_pp, pixel_values,
                                                                         pixel_averages, pixel_var_sum)

        features = pixel_samples[:, feature_indices[:, 0]] - pixel_samples[:, feature_indices[:, 1]]
        ranges = features.min(axis=0), features.max(axis=0)
        thresholds = (ranges[0] + ranges[1]) / 2.0 + np.random.uniform(low=-(ranges[1] - ranges[0]) * 0.1,
                                                                       high=(ranges[1] - ranges[0]) * 0.1)

        bin_ids = np.apply_along_axis(Fern.get_bin, arr=features, axis=1, thresholds=thresholds)
        bin_outputs = self._calc_bin_averages(targets, bin_ids, self.n_features, n_landmarks, self.beta)

        return Fern(n_landmarks, feature_indices, bin_outputs, thresholds)


class Fern:
    def __init__(self, n_landmarks, feature_indices, bins, thresholds):
        self.n_landmarks = n_landmarks
        self.bins = bins
        self.features = feature_indices
        self.thresholds = thresholds
        self.compressed = False
        self.compressed_bins = None
        self.compressed_coeffs = None
        self.Q = None

    @staticmethod
    def get_bin(features, thresholds):
        res = 0
        for i in range(len(features)):
            if features[i] <= thresholds[i]:
                res |= 1 << i
        return res

    def apply(self, pixels, basis=None):
        # Select features from shape-indexed pixels.
        features = pixels[self.features[:, 0]] - pixels[self.features[:, 1]]
        # Get bin of the sample with the given shape-indexed pixels.
        bin_id = self.get_bin(features, self.thresholds)
        if not self.compressed:
            return self.bins[bin_id]
        else:
            return np.sum(basis[self.compressed_bins[bin_id]] * self.compressed_coeffs[bin_id].reshape(self.Q, 1),
                          axis=0)

    # Performs fern compression using orthogonal matching pursuit.
    def compress(self, basis, Q):
        self.compressed = True
        self.Q = Q

        # compressed_bins = []
        n_features = (1 << len(self.features))
        self.compressed_bins = np.zeros((n_features, Q), dtype=int)
        self.compressed_coeffs = np.zeros((n_features, Q))
        for b, current_bin in enumerate(self.bins):
            # compressed_bin = np.zeros((Q, 2))
            residual = current_bin.copy()
            for k in range(Q):
                max_i = np.argmax(basis.dot(residual))

                self.compressed_bins[b][k] = max_i
                compressed_matrix = np.zeros((self.n_landmarks * 2, k + 1))
                for j in range(k + 1):
                    compressed_matrix[:, j] = basis[self.compressed_bins[b][j]]
                compressed_matrix_t = np.transpose(compressed_matrix)

                retval, dst = cv2.solve(compressed_matrix_t.dot(compressed_matrix),
                                        compressed_matrix_t.dot(current_bin), flags=cv2.DECOMP_SVD)
                for j in range(k + 1):
                    self.compressed_coeffs[b][j] = dst[j]
                residual -= compressed_matrix.dot(dst).reshape(2 * self.n_landmarks, )

        self.bins = None
