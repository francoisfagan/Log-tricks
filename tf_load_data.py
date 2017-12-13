"""Defines how all data is to be loaded"""
import numpy as np
from sklearn.datasets import load_svmlight_file


class MNLDataset():
    def __init__(self, x, y):
        self.x = x.todense().A
        self.y = y[:, None].astype(int)
        self.num_examples = x.shape[0]
        self.batch_index = 0

    def next_batch(self, batch_size, proportion_data):
        batch_indices = self.batch_index + np.arange(batch_size)
        # batch_indices = np.mod(batch_indices, self.num_examples)
        # self.batch_index = (self.batch_index + batch_size) % self.num_examples

        max_example_index = int(self.num_examples * proportion_data)
        batch_indices = np.mod(batch_indices, max_example_index)
        self.batch_index = (self.batch_index + batch_size) % max_example_index

        return [self.x[batch_indices, :], self.y[batch_indices, :]]


def load_data(dataset_name):
    print('Loading data')
    file_path = '../UnbiasedSoftmaxData/ProcessedData/' + dataset_name
    train = MNLDataset(*load_svmlight_file(file_path + '_train.txt'))
    test = MNLDataset(*load_svmlight_file(file_path + '_test.txt'))
    test.x = np.pad(test.x, ((0, 0), (0, train.x.shape[1] - test.x.shape[1])), 'constant')

    dim = train.x.shape[1]
    num_classes = int(max(train.y)) + 1
    num_train_points = train.x.shape[0]
    return [train, test, dim, num_classes, num_train_points]
