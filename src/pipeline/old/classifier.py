from abc import ABCMeta, abstractmethod
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Classifier(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        # X: should be (N, D) matrix
        # y: should be (N, ) array
        raise NotImplementedError

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
    def predict(self, X):
        # X: should be (N, D) matrix
        raise NotImplementedError


class LogisticRegressionClassifier(Classifier):
    def __init__(self, **kwargs):
        self.lr = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.lr.fit(X, y)

    def predict(self, X):
        return self.lr.predict(X)

    def _save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(f, self.lr)

    def load(self, path):
        with open(path, 'rb') as f:
            self.lr = pickle.load(f)


class SVCClassifier(Classifier):
    def __init__(self, **kwargs):
        self.svc = SVC(**kwargs)

    def fit(self, X, y):
        self.svc.fit(X, y)

    def predict(self, X):
        return self.svc.predict(X)

    def _save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(f, self.svc)

    def load(self, path):
        with open(path, 'rb') as f:
            self.svc = pickle.load(f)


CLASSIFIER_MAP = dict(
    logistic_regression=LogisticRegressionClassifier,
    svm=SVCClassifier
)


def get_classifier(name, load=None, **kwargs):
    classifier = CLASSIFIER_MAP[name](**kwargs)
    if load is not None:
        classifier.load(load)
    return classifier
