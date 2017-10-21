""" Pipeline for softmax optimization experiments

Based off code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

Author: Francois Fagan
"""

import tensorflow as tf
from tf_load_data import load_data
from tf_run import run
import pickle
from tf_mnl import *
import time

t0 = time.time()
# Parameters (in the future can pass these in from the command line)
learning_rate = 0.01
training_epochs = 1
batch_size = 100
num_epochs_record_cost = 1
num_sampled = 5
num_repeat = 1
cost_name = 'sampled_softmax'  # 'softmax'  # 'nce'  # 'lt' # 'ove' #
dataset_name = 'wiki10'  # 'Bibtex' # 'Eurlex'  # 'mnist'  # 'Delicious' #

train, test, dim, num_classes, num_train_points = load_data(dataset_name)
variables = graph(dim, num_classes, num_train_points, num_sampled)
cost = get_cost(cost_name, num_classes, num_sampled, *variables)
results = run(train, test, num_train_points, cost,
              learning_rate, batch_size, num_epochs_record_cost, num_repeat,
              training_epochs, error, num_classes, cost_name, num_sampled, *variables)
pickle.dump(results, open('./Results/' + cost_name + '_' + dataset_name + '.p', 'wb'))

t1 = time.time()
print('Time', t1 - t0)
