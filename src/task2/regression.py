import numpy as np

import sys
sys.path.append('../task1/')
from wine_dataset import vectorized_data
from knn import cross_validation
from make_figures import PATH, FIG_WITDH, FIG_HEIGHT, FIG_HEIGHT_FLAT, setup_matplotlib


class RidgeRegression:
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        n, d = X.shape
        XTX = np.matmul(X.T,X)
        XTy = np.matmul(X.T,y)
        CId = np.eye(d)/self.C
        self.weight = np.linalg.inv(XTX + CId) @ XTy

    def predict(self, X):
        return X @ self.weight


#TODO: 2c
class RidgeRegressionBias:
    def __init__(self, C=1):
        self.rr = RidgeRegression(C)

    def fit(self, X, y):
        data_size = len(y)
        self.bias = np.mean(y)
        X = np.c_[X, np.ones(data_size)*self.bias]
        self.rr.fit(X,y)

    def predict(self, X):
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)]
        return self.rr.predict(X)


def projection(data):
    #TODO: 2b
    # remember to except the 'point' column
    # it should return an numpy 1d-array
    return data


def transform(data):
    pass


def mean_sqrt_err(clf, X,Y):
    pass


def forward_stepwise_selection(data, points, k=5):
    subset = [] # list of feature name
    #TODO:
    return subset


if __name__ == '__main__':
    # setup_matplotlib()

    # return a dict that values are a list
    data = vectorized_data()
    points = np.array(data.pop('points'))
    print(points)

    #TODO: 2b, 2c
    for col, col_data in data.items():
        train_x = projection(col_data)
        # plot projected data
        clf = RidgeRegression()
        clf.fit(train_x, points)
        # plot clf.predict
        clf = RidgeRegressionBias()
        clf.fit(train_x, points)
        # plot clf.predict

    #TODO: 2d, should be smaller than 6.3
    # print(f'Acc: {cross_validation(RidgeRegression(), transform(data), points,metric=mean_sqrt_err)}')

    #TODO: 2f
    # forward_stepwise_selection(data, points)