""" Pipeline for softmax optimization experiments

Based off code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

Author: Francois Fagan
"""

from load_data import load_data
from run import run
import pickle
import time

t0 = time.time()


# Parameters (in the future can pass these in from the command line)
learning_rate = 0.0001
epochs = 10
batch_size = 1
num_epochs_record = 10
num_sampled = 5
num_repeat = 1
sgd_name = 'softmax'  #'lt'  # 'IS' # 'softmax'  #'sampled_softmax'  # 'ove' #'nce' #
dataset_name = 'Delicious'  #'Eurlex' #'Bibtex'  #  'mnist'  #
run_mnl = True
run_word2vec = False
assert (run_mnl != run_word2vec)

train, test = load_data(dataset_name)
results = run(train, test, learning_rate, batch_size, num_epochs_record, num_repeat, epochs, sgd_name, num_sampled)
# pickle.dump(results, open('./Results/' + sgd_name + '_' + dataset_name + '.p', 'wb'))

t1 = time.time()

total = t1-t0
print(total)