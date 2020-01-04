import pdb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import numpy as np

from compressor import get_compressor
from classifier import get_classifier
from train import load_data


def aggregate_data(data, agg):
    x, y, z = data
    x1, x2 = x.transpose(1, 0, 2)
    if agg == 'minus-abs':
        return np.abs(x1 - x2), y, z
    elif agg == 'mul_minus-abs':
        return np.concatenate([x1 - x2, x1 * x2], axis=-1), y, z
    else:
        raise ValueError("agg cannot be %s" % agg)


class Ours(BaseEstimator, ClassifierMixin):
    def __init__(self, compressor, classifier, agg='minus-abs'):
        self.compressor = compressor
        self.classifier = classifier
        self.agg = agg

    def fit(self, x, y):
        # pdb.set_trace()
        x, y, _ = aggregate_data((x, y, None), self.agg)
        x = self.compressor.fit_transform(x)
        self.classifier.fit(x, y)

    def predict(self, x):
        # pdb.set_trace()
        x, _, _ = aggregate_data((x, None, None), self.agg)
        x = self.compressor.transform(x)
        return self.classifier.predict(x)


class OursSVM(Ours):
    def __init__(self, n_component, **kwargs):
        # pdb.set_trace()
        self.n_component = n_component
        self.kwargs = kwargs
        compressor = get_compressor('pca', n_component)
        classifier = get_classifier('svm', **kwargs)
        super(OursSVM, self).__init__(compressor, classifier)

    def get_params(self, deep=True):
        params = dict(n_component=self.n_component, C=1.0, gamma='scale', shrinking=True, tol=1e-3)
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        # pdb.set_trace()
        if 'n_component' in params:
            self.n_component = params.pop('n_component')
        self.kwargs.update(params)
        self.classifier = get_classifier('svm', **self.kwargs)
        return self


def print_best_score(gsearch, param_test):
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    params = dict(
        # n_component=[64, 128, 256, 512, 1024, 2048, 4096],
        # C=[0.01, 0.1, 1],
        C=[1, ],
        # gamma=['scale', 'auto', 0.01, 0.1, 1, 10],
        # shrinking=[True, False],
        # tol=[1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        tol=[1e-3, ],
    )
    gs = GridSearchCV(OursSVM(1024), params, n_jobs=4)

    folds = load_data('../../data/tenfold_lfw')
    x, y, _ = list(zip(*folds))
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    gs.fit(x, y)
    print_best_score(gs, params)
