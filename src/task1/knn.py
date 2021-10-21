import numpy as np

from dataset import *


def euclidean_distance(a, b):
    raise NotImplementedError('TODO')


class KNN:
    def __init__(self, k=5, dist_function=euclidean_distance):
        self.k = k
        self.dist_function = dist_function

    def fit(self, X, y):
        """
        Train the k-NN classifier.

        :param X: Training inputs. Array of shape (n, ...)
        :param y: Training labels. Array of shape (n,)
        """
        raise NotImplementedError('TODO')

    def predict(self, X):
        """
        Predict labels for new, unseen data.

        :param X: Test data for which to predict labels. Array of shape (n', ..) (same as in fit)
        :return: Labels for all points in X. Array of shape (n',)
        """
        raise NotImplementedError('TODO')


def accuracy(predicted, actual):
    return np.mean(predicted == actual)


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


def main(args):
    # Set up data
    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()
    train_y = np.array(train_y)

    # TODO: Load and evaluate the classifier for different k

    # TODO: Plot results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This script computes cross validation scores for a kNN classifier.')

    parser.add_argument('--folds', '-m', type=int, default=5,
                        help='The number of folds that the data is partitioned in for cross validation.')
    parser.add_argument('--train-data', type=str, default=DEFAULT_ROOT,
                        help='Directory in which the training data and the corresponding labels are located.')
    parser.add_argument('--k', '-k', type=int, default=list(range(1, 11))+[15], nargs='+',
                        help='The k values that should be evaluated.')

    args = parser.parse_args()
    main(args)