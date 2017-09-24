""" Pipeline for softmax optimization experiments

Based off code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

Author: Francois Fagan
"""

import tensorflow as tf
from load_data import load_data
from run import run
import pickle

# Parameters (in the future can pass these in from the command line)
learning_rate = 0.01
train_test_split = 0.7
training_epochs = 100
batch_size = 100
num_epochs_record_cost = 10
num_repeat = 1
cost_name = 'softmax'
dataset_name = 'Delicious' #'Bibtex' #'mnist' #
run_mnl = True
run_word2vec = False
assert (run_mnl != run_word2vec)

if run_mnl:
    from mnl import *
elif run_word2vec:
    from word2vec import *

train, test, dim, num_classes, num_train_points = load_data(dataset_name, train_test_split)
variables = graph(dim, num_classes, num_train_points)
cost = get_cost(cost_name, *variables)
record = run(train, test, num_train_points, *variables, cost,
                                               learning_rate, batch_size, num_epochs_record_cost, num_repeat,
                                               training_epochs, error, num_classes)
pickle.dump(record, open(cost_name + '_' + dataset_name + '.p', 'wb'))