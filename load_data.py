"""Defines how all data is to be loaded"""
import numpy as np
from sklearn.datasets import load_svmlight_file


class MNLDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_examples = x.shape[0]
        self.batch_index = 0

    def next_batch(self, batch_size):
        batch_indices = self.batch_index + np.arange(batch_size)
        batch_indices = np.mod(batch_indices, self.num_examples)
        self.batch_index = (self.batch_index + batch_size) % self.num_examples

        return [self.x[batch_indices, :], self.y[batch_indices, :], batch_indices[:, None]]


def loadLIBSVMdata(file_path, train_test_split):
    # Load the data
    data = load_svmlight_file(file_path, multilabel=True)

    # Separate into x and y
    # Remove data with no y value
    # and if multiple y values, take the first one
    y = data[1]
    y_not_empty = [i for i, y_val in enumerate(y) if y_val != ()]
    y = np.array([y[i][0] for i in y_not_empty])
    x = data[0].toarray()[y_not_empty, :]

    # Find point to split training and test sets
    n_samples = len(y)
    split_point = int(train_test_split * n_samples)

    # Create train and test sets
    train = MNLDataset(x[:split_point, :], y[:split_point])
    test = MNLDataset(x[split_point:, :], y[split_point:])
    return train, test


def load_data(dataset_name, train_test_split):
    if dataset_name == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
        train = MNLDataset(mnist.train.images, mnist.train.labels[:,None]) #
        test = MNLDataset(mnist.test.images, mnist.test.labels[:,None]) #[:,None]
    if dataset_name in {'Bibtex'}:
        file_path = '/Users/francoisfagan/Documents/UnbiasedSoftmaxData/LIBSVM/' + dataset_name + '.txt'
        train, test = loadLIBSVMdata(file_path, train_test_split)

    dim = train.x.shape[1]
    num_classes = int(max(train.y)) + 1 #train.y.shape[1]#
    num_train_points = train.x.shape[0]
    return [train, test, dim, num_classes, num_train_points]

