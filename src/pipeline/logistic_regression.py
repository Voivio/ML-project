import numpy as np
import pdb


class LogisticRegression:
    def __init__(self, lr=0.03, iters=1500, verbose=False):
        self.params = None
        self.lr = lr
        self.iters = iters
        self.verbose = verbose

    def fit(self, X, y):
        params = np.zeros((X.shape[1], 1))
        if self.verbose:
            print("Initial cost = %.4f" % self.compute_cost(X, y, params))

        m = len(y)
        for i in range(self.iters):
            # pdb.set_trace()
            params = params - (self.lr / m) * (X.T @ (self.sigmoid(X @ params) - y.reshape(-1, 1)))
            if self.verbose:
                print("At step %d / %d, cost = %.4f" % (i + 1, self.iters, self.compute_cost(X, y, params)))
        self.params = params

    @staticmethod
    def compute_cost(X, y, theta):
        m = len(y)
        h = LogisticRegression.sigmoid(X @ theta)
        epsilon = 1e-5
        cost = (1 / m) * (((-y).T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
        # pdb.set_trace()
        return float(cost)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return np.round(self.sigmoid(X @ self.params)).squeeze()


CLASSIFIER_MAP = dict(
    logistic_regression=1,  # LogisticRegressionClassifier,
    svm=1,  # SVCClassifier
)


def get_classifier(name, load=None, **kwargs):
    return LogisticRegression(verbose=True)
