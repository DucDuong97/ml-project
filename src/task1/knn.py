import random

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import time
import heapq
import multiprocessing
from functools import partial

from dataset import *


def euclidean_distance(a, b):
    return np.linalg.norm(np.squeeze(a - b))


def manhattan_distance(a, b):
    return np.linalg.norm(np.squeeze(a - b), 1)


def minkows_distance(a, b):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += abs(a[i] - b[i]) ** 3
    return dist ** (1 / 3)
    # return np.linalg.norm(np.squeeze(a - b), 3)


# cpr_count = 0


class Node(object):
    def __init__(self, dist: float, label: int, image):
        self.dist = dist
        self.label = label
        self.image = image

    def __repr__(self):
        return f'Node value: {self.dist}, Label: {self.label}, Image: {self.image}'

    def __lt__(self, other):
        global cpr_count
        # cpr_count += 1
        # use for k-largest-heap
        return self.dist > other.dist


def knn(test, X, k, Y, dist_func, return_neighbor=False):
    dists = []
    # implement top k smallest distance
    for i in range(np.size(Y)):
        node = Node(dist_func(test, X[i]), Y[i], X[i])
        if len(dists) < k:
            heapq.heappush(dists, node)
        else:
            if node < dists[0]: continue
            dists[0] = node
            heapq.heapify(dists)

    # for i in range(D):
    #     dists.append(Node(dist_func(test, X[i]), Y[i]))
    # dists.sort()

    labels = [node.label for node in dists]
    max_label = max(labels, key=labels.count)
    if not return_neighbor:
        return max_label
    else:
        # dists consists of distance, actual label of image and image
        return max_label, dists


def knn_weight(test, X, k, Y, dist_func, inverse_modifier):
    dists = []
    # implement top k smallest distance
    for i in range(np.size(Y)):
        node = Node(dist_func(test, X[i]), Y[i], X[i])
        if len(dists) < k:
            heapq.heappush(dists, node)
        else:
            if node < dists[0]: continue
            dists[0] = node
            heapq.heapify(dists)

    # summing up all labels
    labels = {}
    for i in range(k):
        label = dists[i].label
        if label in labels:
            labels[label] += inverse_modifier / dists[i].dist
        else:
            labels[label] = inverse_modifier / dists[i].dist

    # find highest label
    max_label = None
    max_label_weight = 0
    for k in labels:
        if labels[k] > max_label_weight:
            max_label = k
            max_label_weight = labels[k]
    print("The label prediction is ", max_label, end="\r")
    return max_label


class KNN:
    def __init__(self, k=5, dist_function=euclidean_distance, return_neighbor=False):
        self.k = k
        self.dist_function = dist_function
        self.return_neighbor = return_neighbor

    def fit(self, X, y):
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """
        # pool = multiprocessing.Pool(2)
        # return pool.map(partial(knn, X=self.train_x, k=self.k, Y=self.train_y, dist_func=self.dist_function), X)
        return [knn(x, self.train_x, self.k, self.train_y, self.dist_function, self.return_neighbor) for x in X]


class Weight_KNN:
    def __init__(self, k=5, dist_function=euclidean_distance, inverse_modifier=10):
        self.k = k
        self.dist_function = dist_function
        self.inverse_modifier = inverse_modifier

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
        return [knn_weight(x, self.train_x, self.k, self.train_y, self.dist_function, self.inverse_modifier) for x in X]


def accuracy(clf, X, Y):
    sum = 0
    pred_Y = clf.predict(X)
    for pred_y, y in zip(pred_Y, Y):
        if pred_y == y:
            sum += 1
    D = np.size(Y)
    print(f"Accuracy: {sum}/{D}")
    return sum / D


def get_misclassified(clf, X, Y):
    miss_classified = []
    samples_num = 5

    for (pred_y, neighbors), y, x in zip(clf.predict(X), Y, X):
        if pred_y != y:
            miss_classified.append((x, y, pred_y, neighbors))
            samples_num -= 1
        if samples_num == 0: break

    return miss_classified


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

    tic = time.perf_counter()

    pool = multiprocessing.Pool(m)
    accuracies = pool.map(partial(single_validation, X=X, m=m, Y=Y, clf=clf, metric=metric), range(m))

    # accuracies = []
    # for i in range(m):
    #     accuracies.append(single_validation(i, m, clf, X, Y, metric))

    toc = time.perf_counter()
    print(f"Execute in {toc - tic:0.4f} seconds")
    # print(f"Compare Count: {cpr_count}")

    # calculating average accuracy
    acc = sum(accuracies) / m
    print(f"Acc: {acc}")
    return acc

def cross_misclassify(clf, X, Y, m=5, metric=get_misclassified):
    """
    Performs m-fold cross validation to get misclassified samples.

    :param clf: The classifier which should be tested.
    :param X: The input data. Array of shape (n, ...).
    :param Y: Labels for X. Array of shape (n,).
    :param m: The number of folds.
    :param metric: Metric that should be evaluated on the test fold.
    :return: 5 random misclassified samples
    """
    return single_validation(0, m, clf, X, Y, metric)


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


def print_samples(train_x, train_y):
    unique_labels_set = np.unique(train_y)

    samples = {}
    # Retrieving 4 images for each class
    for i in unique_labels_set:
        samples[i] = np.concatenate(np.where(train_y == i), axis=0)[:4]
    for i in samples:
        imgsList = []
        for j in samples[i]:
            imgsList.append(train_x[j])
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f"class {i}")
        axs = axs.flatten()
        for img, ax in zip(imgsList, axs):
            ax.imshow(np.squeeze(img))
        plt.title(f"Image samples for class {i}")
        plt.show()


def plotMissclassified(miss):
    width = TEXT_WIDTH * TEXT_WIDTH_MUL
    fig = plt.figure(figsize=(width, width*ASPECT))
    for i in range(5):
        fig.add_subplot(5, 6, i * 6 + 1)
        plt.title(f'Actual label {miss[i][1]}\n predicted as {miss[i][2]}')

        plt.imshow(np.squeeze(miss[i][0]))
        for j in range(len(miss[i][3])):
            fig.add_subplot(5, 6, i * 6 + 1 + j + 1)
            plt.title(f'Label: {miss[i][3][j].label}')
            plt.imshow(np.squeeze(miss[i][3][j].image))
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def convolve(x, filter):
    x = np.squeeze(x)
    x_h, x_w = x.shape
    filter = np.flipud(np.fliplr(filter))
    filter_h, filter_w = filter.shape
    res_h = x_h - filter_h + 1
    res_w = x_w - filter_w + 1
    res = np.zeros((res_h, res_w))
    for i in range(res_h):
        for j in range(res_w):
            res[i, j] = (filter * x[i: i + filter_h, j: j + filter_w]).sum()
    return res


def main(args):
    # Set up data
    data_size = 1000
    print(f"data size: {data_size}")

    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()[0:data_size]
    train_y = np.array(train_y)[0:data_size]

    # Plot results
    # cross_validation(knn_set[4], train_x, train_y)

    # a
    # print_samples(train_x, train_y)

    # # TODO: c
    # knn_set = []
    # for i in range(1, 11):
    #     knn = KNN(k=i)
    #     knn_set.append(knn)
    # k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # acc = []
    # for i in range(0, 10):
    #     acc.append(cross_validation(knn_set[i], train_x, train_y))
    # plt.plot(k, acc)
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy for different k in KNN')
    # plt.show()

    # # TODO: e
    best_k = 5  # replace when knowing the best k
    knn_euclid = KNN(best_k, euclidean_distance)
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

    # TODO: g
    # blur_filter = np.ones((3,3)) * 1/9
    # edge_filter = np.array([[-1,0,1],[0,0,0],[1,0,-1]])
    # blur_X = np.array([convolve(i, blur_filter) for i in train_x]) ###
    # edge_X = np.array([convolve(i, edge_filter) for i in train_x]) ###
    # filter = ['no filter','blur','detect edge']
    # acc = [cross_validation(knn_euclid, train_x, train_y),
    #         cross_validation(knn_euclid, blur_X, train_y),
    #         cross_validation(knn_euclid, edge_X, train_y)]
    # plt.plot(filter, acc)
    # plt.xlabel('Filter')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Filters in KNN')
    # plt.show()

    # # TODO: h
    # knn_algo = ['normal KNN', 'weight KNN']
    # acc = [cross_validation(knn_euclid, train_x, train_y),
    #        cross_validation(Weight_KNN(inverse_modifier=10), train_x, train_y)]
    # plt.plot(knn_algo, acc)
    # plt.xlabel('Filter')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Algorithm in KNN')
    # plt.show()

    # TODO: i
    print("________________________________________________________________________________________")
    best_k = 5  # replace when knowing the best k
    knn_euclid = KNN(best_k, euclidean_distance, return_neighbor=True)
    miss = cross_misclassify(knn_euclid, train_x, train_y)
    # knn_manhat = KNN(best_k, manhattan_distance)
    # knn_minkow = KNN(best_k, minkows_distance)
    plotMissclassified(miss)


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
