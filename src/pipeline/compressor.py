from abc import ABCMeta, abstractmethod
import pickle
import os
from sklearn.decomposition import PCA


class Compressor(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X):
        # X: should be (N, D) matrix
        raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        if path is None:
            return
        assert not os.path.exists(path)
        self._save(path)

    @abstractmethod
    def _save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        # X: should be (N, D) matrix
        raise NotImplementedError


class PCACompressor(Compressor):
    # TODO: reimplement or delete when releasing code
    def __init__(self, n_components: int, *args, **kwargs):
        self.n_components = n_components
        self.pca = PCA(n_components, *args, **kwargs)

    def fit(self, X):
        self.pca.fit(X)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

    def transform(self, X):
        return self.pca.transform(X)

    def _save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(f, self.pca)

    def load(self, path):
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
            assert self.pca.n_components_ == self.n_components


COMPRESSOR_MAP = dict(
    pca=PCACompressor
)


def get_compressor(name, load=None, **kwargs):
    compressor = COMPRESSOR_MAP[name](**kwargs)
    if load is not None:
        compressor.load(load)
    return compressor
