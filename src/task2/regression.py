import math

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import sys
import time
sys.path.append('../task1/')
from wine_dataset import vectorized_data, get_wine_reviews_data
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
        X = np.c_[X, np.ones(data_size)]
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
    if data.shape[1] == 1:
        return data
    pca = PCA(n_components=1)
    return pca.fit_transform(data)


def transform(data, filter=[]):
    if len(filter) > 0:
        data = { filter_column : data[filter_column] for filter_column in filter }

    values = list(data.values())
    result = None
    for value in values:
        if result is None:
            result = value
        else:
            result = np.concatenate((result, value), axis=1)
    return np.array(result)


def mean_sqrt_err(clf, X,Y):
    return np.sum(np.power(clf.predict(X) - Y,2)) / len(Y)


def forward_stepwise_selection(data, points, k=5):
    print("Begin forward stepwise selection")
    subset = [] # list of feature names and MSE
    for k in range(k):
        min_mse = math.inf
        significant_feature = None
        for column in data:
            if column not in subset:
                temp_set = subset[:]
                temp_set.append(column)
                print(f'Working with {column}, the current set is {temp_set}')
                current_mse = cross_validation(RidgeRegression(), transform(data, temp_set), points, metric=mean_sqrt_err)
                if current_mse < min_mse:
                    min_mse = current_mse
                    significant_feature = column
        subset.append(significant_feature)
        print(f'k={k}: Feature: {significant_feature} with the mse of {min_mse}')
    return subset



def cross_validation(clf, X, Y, m=5, metric=mean_sqrt_err):
    data_size = np.size(Y)
    fold_size = (int)(data_size / m)

    tic = time.perf_counter()

    accuracies = []
    for i in range(m):
        accuracies.append(single_validation(i, m, clf, X, Y, metric))

    toc = time.perf_counter()
    print(f"Execute in {toc - tic:0.4f} seconds")
    # print(f"Compare Count: {cpr_count}")
    acc = sum(accuracies) / m
    print(f"Acc: {acc}")
    return acc
        

def single_validation(i, m, clf, X, Y, metric):
    data_size = np.size(Y)
    fold_size = (int)(data_size / m)
    include_idx = np.arange(i * fold_size, fold_size + i * fold_size)
    mask = np.array([(i in include_idx) for i in range(data_size)])  ###
    # retrieving the test data for each iteration
    image_test_set = X[mask]
    label_test_set = Y[mask]
    # construct a training_set
    image_train_set = X[~mask]
    label_train_set = Y[~mask]
    # using the training data and evaluate using given metric
    clf.fit(image_train_set, label_train_set)
    return metric(clf, image_test_set, label_test_set)

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
    #     # save plot
    #     ax.legend()
    #     fig.tight_layout()
    #     plt.savefig(os.path.join(PATH, f'2b_{col}.pdf'))
    #     plt.close(fig)

    #TODO: 2d, should be smaller than 6.3
    cross_validation(RidgeRegression(10), transform(data), points,metric=mean_sqrt_err)
    #forward_stepwise_selection(data, points, k=5)

    #TODO: 2f
    # forward_stepwise_selection(data, points)