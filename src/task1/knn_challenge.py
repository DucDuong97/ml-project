import time
import numpy as np

from dataset import *

# Configuration
# Please change this so that KNNClass points to your implementation
import knn
KNNClass = knn.KNN
# KNNargs are passed to the constructor of your class
# Make sure that k=5 and the Euclidean distance is used
KNNargs = dict(k=5)
# Configuration end


def time_predict(predictor, dataset, true_labels):
    # Time prediction
    start = time.perf_counter()
    pred = predictor.predict(dataset)
    end = time.perf_counter()
    time_for_pred = end - start

    # Report Accuracy as well if we get test labels
    if true_labels is None:
        acc = 0
    else:
        acc = np.mean(pred == true_labels)

    return acc, time_for_pred


def main(args):
    # Set up data
    train_x, train_y = get_strange_symbols_train_data(root=args.train_data)
    train_x = train_x.numpy()
    train_y = np.array(train_y)
    test_x, test_y = get_strange_symbols_test_data(root=args.test_data)
    test_x = test_x.numpy()
    test_y = None if test_y is None else np.array(test_y)

    # Load and train predictor
    predictor = KNNClass(**KNNargs)
    predictor.fit(train_x, train_y)

    # Evaluate testing performance
    best_time = float('inf')
    for i in range(args.repetitions):
        acc, pred_time = time_predict(predictor, test_x, test_y)

        if acc < args.threshold:
            print(f'Predictor matched baseline only for {acc*100:.2f}% instead of the required {args.threshold*100:.2f}%.'
                  ' DISQUALIFIED!')
            return

        if pred_time < best_time:
            best_time = pred_time

    print(f'Best prediction time was {best_time:.5f} seconds.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This script evaluates implementations for the KNN challenge.')

    parser.add_argument('--threshold', '-t', type=float, default=0.95,
                        help='This defines the threshold for the accuracy of your KNN classifier w.r.t. our reference '
                             'implementation. Will be 0.95 for the final evaluation.')
    parser.add_argument('--repetitions', '-r', type=int, default=100,
                        help='Number of timed repetitions of the prediction. This will be 100 for the final evaluation.')
    parser.add_argument('--train-data', type=str, default=DEFAULT_ROOT,
                        help='Directory in which the training data and the corresponding labels are located.')
    parser.add_argument('--test-data', type=str, default=os.path.join(DEFAULT_ROOT, 'reference_knn/'),
                        help='Directory in which the test data and the corresponding labels are located.')

    args = parser.parse_args()
    main(args)
