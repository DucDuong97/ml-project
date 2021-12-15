import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
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
    # it should return an numpy of shape (n_samples, 1)
    return np.array([[item[0]] for item in data])

def projectionPCA(data):
    #TODO: 2b
    # remember to except the 'point' column
    # it should return an numpy of shape (n_samples, 1)
    pca = PCA(n_components=1)
    return pca.fit_transform(data)


def transform(data):
    values = list(data.values())
    result = None
    for value in values:
        if result is None:
            result = value
        else:
            print(result.shape, value.shape)
            result = np.concatenate((result, value), axis=1)
    print(values)
    print(result)
    return np.array(result)


def mean_sqrt_err(clf, X,Y):
    # print(clf.predict(X))
    return np.sum(np.power(clf.predict(X) - Y,2)) / len(Y)


def forward_stepwise_selection(data, points, k=5):
    subset = [] # list of feature name
    #TODO:
    return subset


if __name__ == '__main__':
    # setup_matplotlib()

    # return a dict that values are a list
    data = vectorized_data()
    points = np.array(data.pop('points'))


    #TODO: 2b, 2c
    # for col, col_data in data.items():
    #     # set up plot
    #     fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    #     ax = plt.gca()
    #     ax.set_xlabel('Projected Value')
    #     ax.set_ylabel('Points')
    #     ax.grid(linestyle='dashed')
    #     # set up data
    #     train_x = projectionPCA(col_data)
    #     min_x = train_x.min()
    #     max_x = train_x.max()
    #     x = np.reshape(np.linspace(min_x,max_x), (50,1))
    #     # plot projected data
    #     ax.scatter(train_x, points)
    #     # plot clf.predict
    #     clf = RidgeRegression()
    #     clf.fit(train_x, points)
    #     ax.plot(x, clf.predict(x), label='RR')
    #     # plot clf.predict
    #     clf = RidgeRegressionBias()
    #     clf.fit(train_x, points)
    #     ax.plot(x, clf.predict(x), label='RRB')
    #     # plot
    #     fig.tight_layout()
    #     plt.savefig(os.path.join(PATH, f'2b_{col}.pdf'))
    #     plt.close(fig)


    #TODO: 2d, should be smaller than 6.3

    # print(f'Loss: {cross_validation(RidgeRegression(), transform(data), points,metric=mean_sqrt_err)}')

    #TODO: 2f
    # forward_stepwise_selection(data, points)