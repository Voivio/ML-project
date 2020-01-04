from abc import ABCMeta, abstractmethod
import numpy as np


class Compressor(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X):
        # X: should be (N, D) matrix
        raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def transform(self, X):
        # X: should be (N, D) matrix
        raise NotImplementedError


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.

    v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


class PCACompressor(Compressor):
    # TODO: reimplement or delete when releasing code
    def __init__(self, n_components: int):
        self.n_components = n_components
        assert isinstance(n_components, int)

    def fit(self, x):
        assert len(x.shape) == 2  # (N, D) arr
        x = x.astype(np.float)

        # Handle svd_solver
        if max(x.shape) <= 500:
            return self._fit_full(x)
        elif self.n_components >= 1 and self.n_components < 0.8 * min(x.shape):
            # return self._fit_randomized(x)
            import pdb
            pdb.set_trace()
            return self._fit_full(x)
        else:
            return self._fit_full(x)

    def _fit_full(self, X):
        """Fit the model by computing full SVD on X"""
        self.solver = 'full'
        n_samples, n_features = X.shape
        n_components = self.n_components

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, V = np.linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V

    def _fit_randomized(self, X):
        """Fit the model by computing truncated SVD (by ARPACK or randomized)
        on X
        """
        self.solver = 'randomized'
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        # sign flipping is done inside
        U, S, V = randomized_svd(X, n_components=n_components,
                                 n_iter=self.iterated_power,
                                 flip_sign=True,
                                 random_state=random_state)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = V
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio_ = \
            self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() -
                                    self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0.

        return U, S, V

    def fit_transform(self, X):
        self.fit(X)
        return self.pca.fit_transform(X)

    def transform(self, X):
        return self.pca.transform(X)


COMPRESSOR_MAP = dict(
    pca=PCACompressor
)


def get_compressor(name, n_component, load=None, **kwargs):
    compressor = COMPRESSOR_MAP[name](n_component, **kwargs)
    if load is not None:
        compressor.load(load)
    return compressor
