import random

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import heapq
import multiprocessing
from itertools import repeat
from functools import partial

from dataset import *


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)


def manhattan_distance(a, b):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += abs(a[i] - b[i])
    return dist


def minkows_distance(a, b):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += abs(a[i] - b[i]) ** 3
    return dist ** (1 / 3)

# cpr_count = 0

class Node(object):
    def __init__(self, dist: float, label: int):
        self.dist = dist
        self.label = label

    def __repr__(self):
        return f'Node value: {self.dist}, Label: {self.label}'

    def __lt__(self, other):
        global cpr_count
        # cpr_count += 1
        # use for k-largest-heap
        return self.dist > other.dist


def knn(test, X, k, Y, dist_func):
    dists = []
    # implement top k smallest distance
    for i in range(np.size(Y)):
        node = Node(dist_func(test, X[i]), Y[i])
        if len(dists) < k:
            heapq.heappush(dists, node)
        else:
            if node < dists[0]: continue
            dists[0] = node
            heapq.heapify(dists)

    # for i in range(D):
    #     dists.append(Node(dist_func(test, X[i]), Y[i]))
    # dists.sort()

    # counting labels
    labels = {}
    for i in range(k):
        label = dists[i].label
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

    # find highest label
    max_label = None
    max_label_amount = 0
    for k in labels:
        if labels[k] > max_label_amount:
            max_label = k
            max_label_amount = labels[k]
    print("The label prediction is ", max_label, end="\r")
    return max_label


class KNN:
    def __init__(self, k=5, dist_function=euclidean_distance):
        self.k = k
        self.dist_function = dist_function

    def fit(self, X, y):
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """
        print(f"k: {self.k}")
        res = []
        # pool = multiprocessing.Pool(4)

        # res = pool.starmap(knn, zip(X, repeat(self.train_x), repeat(self.k), repeat(self.train_y), repeat(self.dist_function)))
        
        # res = pool.map(partial(knn, X=self.train_x, k=self.k, Y=self.train_y, dist_func=self.dist_function), X)
        
        for x in X:
            res.append(knn(x, self.train_x, self.k, self.train_y, self.dist_function))
        
        print()
        return res

class Weight_KNN:
    def __init__(self, k=5, dist_function=euclidean_distance):
        self.k = k
        self.dist_function = dist_function

    def fit(self, X, y):
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        # TODO: b
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """
        raise NotImplementedError('TODO')


def accuracy(clf, X, Y):
    sum = 0
    pred_Y = clf.predict(X)
    for pred_y, y in zip(pred_Y, Y):
        if (pred_y == y):
            sum += 1
    D = np.size(Y)
    print(f"Accuracy: {sum}/{D}")
    return sum / D


def cross_validation(clf, X, Y, m=5, metric=accuracy):
    """
    Performs m-fold cross validation.

    :param clf: The classifier which should be tested.
    :param X: The input data. Array of shape (n, ...).
    :param Y: Labels for X. Array of shape (n,).
    :param m: The number of folds.
    :param metric: Metric that should be evaluated on the test fold.
    :return: The average metric over all m folds.
    """
    data_size = np.size(Y)
    fold_size = (int) (data_size / m)

    tic = time.perf_counter()

    pool = multiprocessing.Pool(m)
    accuracies = pool.map(partial(single_validation, X=X, m=m, Y=Y, clf=clf, metric=metric), range(m))

    # for i in range(m):
    #     accuracies.append(single_validation(i, m, clf, X, Y, metric))

    toc = time.perf_counter()
    print()
    print(f"Execute in {toc - tic:0.4f} seconds")
    # print(f"Compare Count: {cpr_count}")

    # calculating average accuracy
    acc = sum(accuracies) / len(accuracies)
    print(f"Acc: {acc}")
    print(f"--------------------------------------")
    return acc

def single_validation(i, m, clf, X, Y, metric=accuracy):
        data_size = np.size(Y)
        fold_size = (int) (data_size / m)
        include_idx = np.arange(i*fold_size,fold_size + i*fold_size)
        mask = np.array([(i in include_idx) for i in range(data_size)])     ###
        # retrieving the test data for each iteration
        image_test_set = X[mask]
        label_test_set = Y[mask]

        # construct a training_set
        image_train_set = X[~mask]
        label_train_set = Y[~mask]

        # using the training data and evaluate using given metric
        clf.fit(image_train_set, label_train_set)
        return metric(clf, image_test_set, label_test_set)


def print_samples(train_x, train_y):
    unique_y = np.unique(train_y)
    label_y = unique_y[2]

    imgsList = []
    for x, y in zip(train_x, train_y):
        if y == label_y:
            imgsList.append(x)
        if len(imgsList) == 16:
            break

    imgs = np.array(imgsList)
    _, axs = plt.subplots(4, 4, figsize=(8, 6))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(np.squeeze(img))
    plt.show()


def convolution(x, filter):
    squeeze_x = np.squeeze(x)
    # TODO


def main(args):
    # Set up data
    data_size = 8000
    print(f"data size: {data_size}")

    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()[0:data_size]
    train_y = np.array(train_y)[0:data_size]

    # test_x, test_y = get_strange_symbols_test_data(root=args.test_data)
    # test_x = test_x.numpy()
    # test_y = np.array(test_y)

    # Load and evaluate the classifier for different k
    knn_set = []
    for i in range(1, 11):
        knn = KNN(k=i)
        knn_set.append(knn)

    # Plot results

    cross_validation(knn_set[4], train_x, train_y)

    # TODO: a
    # print_samples(train_x, train_y)

    # TODO: c
    # k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # acc = []
    # for i in range(0, 10):
    #     acc.append(cross_validation(knn_set[i], train_x, train_y))
    # plt.plot(k, acc)
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy for different k in KNN')
    # plt.show()

    # TODO: e
    # best_k = 5 # replace when knowing the best k
    # knn_euclid = KNN(best_k, euclidean_distance)
    # knn_manhat = KNN(best_k, manhattan_distance)
    # knn_minkow = KNN(best_k, minkows_distance)

    # dist = ['knn_euclid','knn_manhat','knn_minkow']
    # acc = [cross_validation(knn_euclid, train_x, train_y),
    #        cross_validation(knn_manhat, train_x, train_y),
    #        cross_validation(knn_minkow, train_x, train_y)]
    # plt.plot(dist, acc)
    # plt.xlabel('Distance Function')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Distance Function in KNN')
    # plt.show()

    #TODO: g
    # blur_filter = []
    # edge_filter = []
    # blur_X = map(lambda x: convolution(x, blur_filter),train_x)
    # edge_X = map(lambda x: convolution(x, edge_filter),train_x)
    # filter = ['no filter','blur','detect edge']
    # acc = [cross_validation(knn_euclid, train_x, train_y),
    #         cross_validation(knn_euclid, blur_X, train_y),
    #         cross_validation(knn_euclid, edge_X, train_y)]
    # plt.plot(filter, acc)
    # plt.xlabel('Filter')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Filters in KNN')
    # plt.show()

    # TODO: h
    # knn_algo = ['normal KNN','weight KNN']
    # acc = [cross_validation(knn_euclid, train_x, train_y),
    #         cross_validation(Weight_KNN(), train_x, train_y)]
    # plt.plot(knn_algo, acc)
    # plt.xlabel('Filter')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Algorithsm in KNN')
    # plt.show()

    # TODO: i

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='This script computes cross validation scores for a kNN classifier.')

    parser.add_argument('--folds', '-m', type=int, default=5,
                        help='The number of folds that the data is partitioned in for cross validation.')
    parser.add_argument('--train-data', type=str, default=DEFAULT_ROOT,
                        help='Directory in which the training data and the corresponding labels are located.')
    parser.add_argument('--k', '-k', type=int, default=list(range(1, 11)) + [15], nargs='+',
                        help='The k values that should be evaluated.')

    args = parser.parse_args()
    main(args)
