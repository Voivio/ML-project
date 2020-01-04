import numpy as np
import pdb
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr=0.03, eps=1e-4, iters=1500, verbose=False):
        self.params = None
        self.lr = lr
        self.eps = eps
        self.iters = iters
        self.verbose = verbose

    def fit(self, X, y):
        return self.fit_adam(X, y)
        params = np.zeros((X.shape[1], 1))
        costs = [self.compute_cost(X, y, params), ]
        if self.verbose:
            print("Initial cost = %.4f" % costs[-1])

        m = len(y)
        for i in range(self.iters):
            # pdb.set_trace()
            params = params - (self.lr / m) * (X.T @ (self.sigmoid(X @ params) - y.reshape(-1, 1)))
            costs.append(self.compute_cost(X, y, params))
            if self.verbose:
                print("At step %d / %d, cost = %.4f" % (i + 1, self.iters, costs[-1]), end='\r')
            if np.abs(costs[-1] - costs[-2]) < self.eps:
                break
        self.params = params

        if self.verbose:
            print()
            plt.plot(costs)
            plt.show()

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

    def adam(self, params, vs, sqrs, lr, batch_size, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        for param, v, sqr in zip(params, vs, sqrs):
            g = param.grad / batch_size

            v[:] = beta1 * v + (1. - beta1) * g
            sqr[:] = beta2 * sqr + (1. - beta2) * np.square(g)

            v_bias_corr = v / (1. - beta1 ** t)
            sqr_bias_corr = sqr / (1. - beta2 ** t)

            div = lr * v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
            param[:] = param - div

    def fit_adam(self, X, y):
        params = np.zeros((X.shape[1], 1))
        costs = [self.compute_cost(X, y, params), ]
        if self.verbose:
            print("Initial cost = %.4f" % costs[-1])

        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        v = np.zeros(params.shape, dtype=params.dtype)
        sqr = np.zeros(params.shape, dtype=params.dtype)

        for i in range(self.iters):
            # pdb.set_trace()
            g = X.T @ (self.sigmoid(X @ params) - y.reshape(-1, 1))
            v = beta1 * v + (1. - beta1) * g
            sqr = beta2 * sqr + (1. - beta2) * np.square(g)

            t = i + 1
            v_bias_corr = v / (1. - beta1 ** t)
            sqr_bias_corr = sqr / (1. - beta2 ** t)

            div = self.lr * v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
            params = params - div

            costs.append(self.compute_cost(X, y, params))
            if self.verbose:
                print("At step %d / %d, cost = %.4f" % (i + 1, self.iters, costs[-1]), end='\r')
            if np.abs(costs[-1] - costs[-2]) < self.eps:
                break
        self.params = params

        if self.verbose:
            print()
            plt.plot(costs)
            plt.show()
