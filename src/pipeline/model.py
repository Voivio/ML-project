import numpy as np
from sklearn import metrics


class Model:
    AGG_CHOICES = ['minus-abs', 'mul_minus-abs', 'mul', 'concat']

    def __init__(self, agg, compressor, classifier, default_option=None):
        self.agg = agg
        assert agg in self.AGG_CHOICES
        self.compressor = compressor
        self.classifier = classifier
        self.default_option = default_option

    def aggregate(self, data):
        x, y, z = data
        x1, x2 = x.transpose(1, 0, 2)
        if self.agg == 'minus-abs':
            return np.abs(x1 - x2), y, z
        elif self.agg == 'mul':
            return x1 * x2, y, z
        elif self.agg == 'mul_minus-abs':
            return np.concatenate([x1 - x2, x1 * x2], axis=-1), y, z
        elif self.agg == 'concat':
            if y is not None:
                return np.concatenate([np.concatenate([x1, x2], axis=-1),
                                       np.concatenate([x2, x1], axis=-1)], axis=0),\
                       np.concatenate([y, y], axis=0), None if z is None else np.concatenate([z, z], axis=0)
            else:
                return np.concatenate([x1, x2], axis=-1), y, z
        else:
            raise ValueError("agg cannot be %s" % self.agg)

    def train(self, train):
        train_x, train_y, train_z = self.aggregate(train)
        train_x = self.compressor.fit_transform(train_x)
        self.classifier.fit(train_x, train_y)
        self.default_option = np.argmax([(train_z == 0).sum(), (train_z == 1).sum()])

    def infer(self, x):
        x, _, _ = self.aggregate((x, None, None))
        x = self.compressor.transform(x)
        return self.classifier.predict(x)

    def evaluate(self, data):
        x, y, z = data
        pred = self.infer(x)
        print("\tw/o invalid data:")
        print("\t\tAcc score  = %.2f" % (metrics.accuracy_score(y, pred) * 100))
        print("\t\tF1 score   = %.2f" % (metrics.f1_score(y, pred) * 100))
        y = np.concatenate([y, z], axis=0)
        pred = np.concatenate([pred, np.repeat(self.default_option, z.shape)], axis=0)
        print("\twith invalid data:")
        acc = metrics.accuracy_score(y, pred) * 100
        f1 = metrics.f1_score(y, pred) * 100
        print("\t\tAcc score  = %.2f" % acc)
        print("\t\tF1 score   = %.2f" % f1)
        return acc, f1
