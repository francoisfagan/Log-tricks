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
learning_rate = 0.0001
train_test_split = 0.7
training_epochs = 1000
batch_size = 100
num_epochs_record_cost = 10
num_sampled = 5
num_repeat = 5
cost_name = 'softmax_IS'  #'lt'  # 'IS' # 'softmax'  #'sampled_softmax'  # 'ove' #'nce' #
dataset_name = 'Bibtex'  # 'Delicious'  # 'mnist'  # 'Euler' #
run_mnl = True
run_word2vec = False
assert (run_mnl != run_word2vec)

if run_mnl:
    from mnl import *
elif run_word2vec:
    from word2vec import *

train, test, dim, num_classes, num_train_points = load_data(dataset_name, train_test_split)
variables = graph(dim, num_classes, num_train_points, num_sampled)
cost = get_cost(cost_name, num_classes, num_sampled, *variables)
results = run(train, test, num_train_points, cost,
              learning_rate, batch_size, num_epochs_record_cost, num_repeat,
              training_epochs, error, num_classes, cost_name, num_sampled, *variables)
pickle.dump(results, open('./Results/' + cost_name + '_' + dataset_name + '.p', 'wb'))
