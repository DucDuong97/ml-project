import random

import numpy as np
from scipy.spatial import distance
import torch
import math
import matplotlib.pyplot as plt
import time
import heapq
import multiprocessing
from functools import partial

from dataset import *

# Path to which figures will be saved
PATH = '../../report/task1/figures'
# Width of the text area in our document. Given in inches
TEXT_WIDTH = 5.78851
# Multiplicators for the figure's width
TEXT_WIDTH_MUL = 0.7
TEXT_WIDTH_MUL_WIDE = 0.9
# Aspect ratio of figures
ASPECT = 3. / 4.

FIG_WITDH = TEXT_WIDTH * TEXT_WIDTH_MUL
FIG_HEIGHT = FIG_WITDH * ASPECT
FIG_HEIGHT_FLAT = FIG_HEIGHT * 0.45

# Font sizes
SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

def setup_matplotlib():
    # Use LaTeX to typeset all text in the figure
    # This obviously needs a working LaTeX installation on the system
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': SMALL_SIZE,
        'axes.titlesize': SMALL_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': MEDIUM_SIZE,
        'text.usetex': True,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'text.latex.preamble': ['\\usepackage{amsmath}\n'
                                '\\usepackage{amssymb}']
    })

#####################################################################

# distance function

def euclidean_distance(a, b):
    return np.linalg.norm(np.squeeze(a - b))


def manhattan_distance(a, b):
    return np.linalg.norm(np.squeeze(a - b), 1)


def minkowki_distance(p, q):
    return distance.minkowski(p.flatten(), q.flatten())

#####################################################################

# help function

class DistMatrix(object):
    def __init__(self, X, dist_func, filter=None):
        print("creating dist matrix")
        tic = time.perf_counter()
        len_x = len(X)
        self.dist_func = dist_func
        self.X = X
        self.dists = np.empty((len_x, len_x))

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        if filter is not None:
            self.X = pool.map(partial(convolve, filter=filter), X)
        self.dists = pool.map(self.worker, range(len_x))

        # for i in range(len_x):
        #     for j in range(i):
        #         dist = dist_func(X[i],X[j])
        #         self.dists[i,j] = dist

        toc = time.perf_counter()

        print(f"Execute in {toc - tic:0.4f} seconds")

    def worker(self, i):
        res = np.empty(len(self.X))
        for j in range(i+1):
            res[j] = self.dist_func(self.X[i],self.X[j])
        return res

    def dist(self, d1, d2):
        return self.dists[d1][d2] if d1 >= d2 else self.dists[d2][d1]

    def get(self, i):
        return self.X[i]


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
        node = Node(dist_func.dist(test, X[i]), Y[i], dist_func.get(X[i]))
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
        return max_label, dists, dist_func.get(test)


def knn_weight(test, X, k, Y, dist_func, inverse_modifier):
    dists = []
    # implement top k smallest distance
    for i in range(np.size(Y)):
        node = Node(dist_func.dist(test, X[i]), Y[i], X[i])
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


#####################################################################

# Classes

class KNN:
    def __init__(self, k=5, dist_function=euclidean_distance, filter=None, return_neighbor=False):
        self.k = k
        self.dist_function = dist_function
        self.return_neighbor = return_neighbor
        self.filter = filter

    def fit(self, X, y):
        if self.filter is not None:
            X = np.array([convolve(i, self.filter) for i in X]) 
        self.train_x = X
        self.train_y = y

    def predict(self, X):
        if self.filter is not None:
            X = np.array([convolve(i, self.filter) for i in X]) 
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

#####################################################################

# Metric function

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

    for (pred_y, neighbors, x), y in zip(clf.predict(X), Y):
        if pred_y != y:
            miss_classified.append((x, y, pred_y, neighbors))
            samples_num -= 1
        if samples_num == 0: break

    return miss_classified


#####################################################################

# K-fold Cross Validation

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
    print()
    print(f"{m}-fold validation")
    print(f"k: {clf.k}")

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
    print('--------------------------')
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


#####################################################################

# Plot function

def plot_samples(sample_x, sample_y, amount):
    unique_labels_set = np.unique(sample_y)

    samples = {}
    # Retrieving 4 images for each class
    for i in unique_labels_set:
        samples[i] = np.concatenate(np.where(sample_y == i), axis=0)[:amount]
    for i in samples:
        imgsList = []
        for j in samples[i]:
            imgsList.append(sample_x[j])
        fig, axs = plt.subplots(1, amount, figsize=(FIG_WITDH, FIG_HEIGHT_FLAT))
        fig.suptitle(f"Class {i}")
        for img, ax in zip(imgsList, axs):
            ax.imshow(np.squeeze(img))
        plt.savefig(os.path.join(PATH, f'1a_sample_class_{i}.pdf'))
        plt.close(fig)


def plotMissclassified(miss):
    amount = len(miss)
    if amount ==0: return
    for i in range(amount):
        img, actual, pred, neighbors = miss[i]
        neigh_num = len(neighbors)
        fig, axs = plt.subplots(1, neigh_num + 1, figsize=(FIG_WITDH, FIG_HEIGHT_FLAT))
        axs[0].set_title(f'Actual: {actual}\n predicted: {pred}')
        axs[0].imshow(np.squeeze(img))

        for j in range(neigh_num):
            axs[j+1].set_title(f'Lbl: {neighbors[j].label}')
            axs[j+1].imshow(np.squeeze(neighbors[j].image))
        
        plt.savefig(os.path.join(PATH, f'1i_miss_classified_{i}.pdf'))
        plt.close(fig)


#####################################################################

# Mainnnnn

def main(args):
    # Set up plot
    setup_matplotlib()

    # Set up data
    data_size = 1000
    print(f"data size: {data_size}")

    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()[0:data_size]
    train_y = np.array(train_y)[0:data_size]

    dist_matrix_euclid = DistMatrix(train_x, euclidean_distance)
    train_x_idx = np.arange(data_size)
    
    # Plot results
    # cross_validation(KNN(), train_x_idx, train_y)



    # TODO: a
    # amount = 4
    # sample_x = train_x[:1000]
    # sample_y = train_y[:1000]
    # plot_samples(sample_x, sample_y, amount)



    # TODO: c
    # k = range(1,11)
    # acc = [cross_validation(KNN(i+1,dist_function=dist_matrix_euclid), train_x_idx, train_y) for i in range(10)]

    # fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    # plt.plot(k, acc)
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    # plt.title('Accuracy for different k in KNN')
    # fig.tight_layout()
    # plt.savefig(os.path.join(PATH, f'1c_knn_acc_k.pdf'))
    # plt.close(fig)



    # # TODO: e
    # best_k = 5  # replace when knowing the best k
    # dist_matrix_manhattan = DistMatrix(train_x, manhattan_distance)
    # dist_matrix_minkowski = DistMatrix(train_x, minkowki_distance)
    # knn_euclid = KNN(best_k, dist_function=dist_matrix_euclid)
    # knn_manhat = KNN(best_k, dist_function=dist_matrix_manhattan)
    # knn_kl = KNN(best_k, dist_function=dist_matrix_minkowski)

    # dist = ['Euclid','Manhattan','Minkowski']
    # acc = [cross_validation(knn_euclid, train_x_idx, train_y),
    #        cross_validation(knn_manhat, train_x_idx, train_y),
    #        cross_validation(knn_kl, train_x_idx, train_y)]

    # fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    # plt.plot(dist, acc)
    # plt.xlabel('Distance Function')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Distance Function in KNN')
    # fig.tight_layout()
    # plt.savefig(os.path.join(PATH, f'1e_knn_acc_dist.pdf'))
    # plt.close(fig)



    # TODO: g
    # dist_matrix_euclid_blur = DistMatrix(train_x, euclidean_distance, filter=np.ones((3,3)) * 1/9)
    # dist_matrix_euclid_edge = DistMatrix(train_x, euclidean_distance, filter=np.array([[-1,0,1],[0,0,0],[1,0,-1]]))
    # knn_non_filter  = KNN(dist_function=dist_matrix_euclid)
    # knn_blur_filter = KNN(dist_function=dist_matrix_euclid_blur)
    # knn_edge_filter = KNN(dist_function=dist_matrix_euclid_edge)

    # filter = ['no filter','blur','detect edge']
    # acc = [cross_validation(knn_non_filter, train_x_idx, train_y),
    #         cross_validation(knn_blur_filter, train_x_idx, train_y),
    #         cross_validation(knn_edge_filter, train_x_idx, train_y)]

    # fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    # plt.plot(filter, acc)
    # plt.xlabel('Filter')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for different Filters in KNN')
    # fig.tight_layout()
    # plt.savefig(os.path.join(PATH, f'1g_knn_acc_filter.pdf'))
    # plt.close(fig)



    # # TODO: h
    knn_algo = ['normal KNN', 'weight KNN']
    acc = [cross_validation(KNN(dist_function=dist_matrix_euclid), train_x_idx, train_y),
           cross_validation(Weight_KNN(inverse_modifier=10,dist_function=dist_matrix_euclid), train_x_idx, train_y)]

    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.plot(knn_algo, acc)
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different Algorithm in KNN')
    fig.tight_layout()
    plt.savefig(os.path.join(PATH, f'1h_knn_acc_algo.pdf'))
    plt.close(fig)



    # TODO: i
    # print("________________________________________________________________________________________")
    # miss_classified_k = 4  # Best for printing
    # knn_euclid = KNN(miss_classified_k, dist_function=dist_matrix_euclid, return_neighbor=True)
    # miss = single_validation(0, 4, knn_euclid, train_x_idx, train_y, get_misclassified)
    # plotMissclassified(miss)
    # knn_manhat = KNN(best_k, manhattan_distance)
    # knn_minkow = KNN(best_k, minkows_distance)



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
