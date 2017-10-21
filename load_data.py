"""Defines how all data is to be loaded"""
import numpy as np
from sklearn.datasets import load_svmlight_file
import pickle
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, find


class MNLDataset():
    def __init__(self, file_name):
        x, y = load_svmlight_file(file_name + '.txt')
        self.x_sparse = x  # Dimensions [num_examples] x [dim]
        self.y = y.astype(int)  # Dimension [num_examples]
        self.num_examples = x.shape[0]
        self.batch_index = 0

        # It will be convenient to express x in a tuple of the form
        # self.x = tuple((x_row, *find(x_row)) for x_row in x)
        # We can do this directly, or use the pre-computed pickled value
        self.x = list(pickle.load(open(file_name + '_tuple.p', 'rb')))

    def next_batch(self, batch_size):
        batch_indices = self.batch_index + np.arange(batch_size)
        batch_indices = np.mod(batch_indices, self.num_examples)
        self.batch_index = (self.batch_index + batch_size) % self.num_examples

        return [self.x[batch_indices[0]], self.y[batch_indices], batch_indices]


def load_data(dataset_name):
    print('Loading data')
    file_path = '../UnbiasedSoftmaxData/ProcessedData/' + dataset_name
    train = MNLDataset(file_path + '_train')
    test = MNLDataset(file_path + '_test')
    test.x_sparse._shape = (test.x_sparse._shape[0], train.x_sparse.shape[1])
    return [train, test]
