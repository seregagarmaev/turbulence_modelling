import numpy as np
from modules.tbdt_v8 import TBDT


class SquareLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class TBGB:
    def __init__(self, n_trees, learning_rate=10**(-2), max_levels=100, min_samples_leaf=1, verbose=True, splitting_features='all',
                 tree_filename='data/TBGB/TREE_GB_%i', regularization=True, regularization_lambda=0.01,
                 optim_split=True, optim_threshold=1000):
        self.n_trees = n_trees
        self.init_values = None
        self.loss = SquareLoss()
        self.learning_rate = learning_rate
        self.trees = {}

        # properties inherited from TBDT
        self.max_levels = max_levels
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.splitting_features = splitting_features
        self.tree_filename = tree_filename
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        self.optim_split = optim_split
        self.optim_threshold = optim_threshold

    def randomSampling(self, X, Y, TB, size_out=10000, replace=True):
        """
        Take random samples with or without replacement from data,
        N_samples = fraction*length(array)
        """

        # samples from the columns:
        idx = np.random.choice(X.shape[1], int(size_out), replace=replace)

        X_out = X[:, idx]
        Y_out = Y[:, idx]
        TB_out = TB[:, :, idx]

        return X_out, Y_out, TB_out

    def fit(self, X, Y, TB):
        self.init_values = Y.mean(axis=1)
        y_pred = (np.ones(Y.T.shape) * self.init_values).T
        y_pred = np.zeros(Y.shape)
        for i in range(self.n_trees):
            gradient = self.loss.gradient(Y, y_pred)
            X_sampled, Y_sampled, TB_sampled = self.randomSampling(X, gradient, TB)

            tree_filename = (self.tree_filename % i)
            tbdt = TBDT(max_levels=self.max_levels, min_samples_leaf=self.min_samples_leaf,
                        regularization=self.regularization, regularization_lambda=self.regularization_lambda,
                        splitting_features=self.splitting_features, tree_filename=tree_filename,
                        verbose=self.verbose, optim_split=self.optim_split, optim_threshold=self.optim_threshold)
            tree = tbdt.fit(X_sampled, Y_sampled, TB_sampled)
            update, g_update = tbdt.predict(X, TB, tree)
            y_pred -= self.learning_rate * update
            self.trees[i] = tree
        return self.trees

    def predict(self, X_test, TB_test, forest):
        y_pred = (np.ones((TB_test.shape[2], TB_test.shape[0])) * self.init_values).T
        y_pred = np.zeros((TB_test.shape[0], X_test.shape[1]))
        for i in range(len(forest)):
            tree_filename = (self.tree_filename % i)
            tbdt = TBDT(max_levels=self.max_levels, min_samples_leaf=self.min_samples_leaf,
                        regularization=self.regularization, regularization_lambda=self.regularization_lambda,
                        splitting_features=self.splitting_features, tree_filename=tree_filename,
                        verbose=self.verbose, optim_split=self.optim_split, optim_threshold=self.optim_threshold)
            update, g_update = tbdt.predict(X_test, TB_test, self.trees[i])
            y_pred -= self.learning_rate * update
        return y_pred