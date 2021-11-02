import numpy as np
import math
import matplotlib.pyplot as plt

from dataset import *


def euclidean_distance(a, b):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return math.sqrt(dist)

def manhattan_distance(a, b):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += abs(a[i] - b[i])
    return dist

def minkows_distance(a, b, root):
    dist = 0.0
    a = a.flatten()
    b = b.flatten()
    for i in range(len(a)):
        dist += abs(a[i] - b[i]) ** root
    return dist ** (1/root)


def find_neighbors(test, X, k):
    """
    Find k nearest neighbors of point test

    :param test: Calculating point
    :param X:
    :param k: Number of nearest neighbors need to find
    :return: List of k nearest neighbors (may contain the test point itself)
    """
    dists = []
    # X = np.array(X).tolist()
    for i in X:
        dists.append((i, euclidean_distance(test, i)))
    dists.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append((dists[i][0], dists[i][1]))
        print("Nearest neighbor with the distance ", dists[i][1])
    return neighbors


def label(neighbors, Y):
    print(np.size(Y))
    for i in neighbors:
        print(Y[np.where(neighbors == i)])


class KNN:
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
    raise NotImplementedError('TODO')


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

def main(args):
    # Set up data
    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()
    train_y = np.array(train_y)

    # test_x, test_y = get_strange_symbols_test_data(root=args.test_data)
    # test_x = test_x.numpy()
    # test_y = np.array(test_y)

    # Load and evaluate the classifier for different k
    knn_set = []
    for i in range(1,11):
        knn = KNN(k=i)
        knn_set.append(knn)

    # Plot results
    # TODO: a
    print_samples(train_x, train_y)

    # TODO: c
    k = [1,2,3,4,5,6,7,8,9,10]
    acc = []
    for i in range(1,11):
        acc.append(cross_validation(knn_set[i], train_x, train_y))
    plt.plot(k, acc)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuracy for different k in KNN')
    plt.show()

    # TODO: d
    best_k = 5 # replace when knowing the best k
    knn_euclid = KNN(best_k, euclidean_distance)
    knn_manhat = KNN(best_k, manhattan_distance)
    knn_minkow = KNN(best_k, minkows_distance)

    dist = [knn_euclid,knn_manhat,knn_minkow]
    acc = [cross_validation(knn_euclid, train_x, train_y),
            cross_validation(knn_manhat, train_x, train_y),
            cross_validation(knn_minkow, train_x, train_y)]
    plt.plot(dist, acc)
    plt.xlabel('Distance Function')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different Distance Function in KNN')
    plt.show()

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
