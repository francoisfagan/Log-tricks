"""Defines how all data is to be loaded"""
import numpy as np
from sklearn.datasets import load_svmlight_file


class MNLDataset():
    def __init__(self, x, y):
        self.x = x  # Dimensions [num_examples] x [dim]
        self.y = y.astype(int)  # Dimension [num_examples]
        self.num_examples = x.shape[0]
        self.batch_index = 0

    def next_batch(self, batch_size):
        batch_indices = self.batch_index + np.arange(batch_size)
        batch_indices = np.mod(batch_indices, self.num_examples)
        self.batch_index = (self.batch_index + batch_size) % self.num_examples

        return [self.x[batch_indices, :], self.y[batch_indices], batch_indices]


def load_data(dataset_name):
    print('Loading data')
    file_path = '../UnbiasedSoftmaxData/ProcessedData/' + dataset_name
    train = MNLDataset(*load_svmlight_file(file_path + '_train.txt'))
    test = MNLDataset(*load_svmlight_file(file_path + '_test.txt'))
    return [train, test]

