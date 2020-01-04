import numpy as np

from svd_util import randomized_svd


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        assert isinstance(n_components, int)

    def fit(self, X):  # (N, D) arr
        X = X.astype(np.float)
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # sign flipping is done inside
        U, S, V = randomized_svd(X, n_components=self.n_components, flip_sign=True)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = V
        self.n_components_ = self.n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio_ = \
            self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() -
                                    self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - self.n_components
        else:
            self.noise_variance_ = 0.

        return self

    def transform(self, X):
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed
