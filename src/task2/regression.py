import numpy as np


class RidgeRegression:
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        raise NotImplementedError('TODO')

    def predict(self, X):
        raise NotImplementedError('TODO')