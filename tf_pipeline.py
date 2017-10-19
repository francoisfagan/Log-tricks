""" Pipeline for softmax optimization experiments

Based off code from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

Author: Francois Fagan
"""

from tf_run import run
import pickle
import time

t0 = time.time()
# Parameters (in the future can pass these in from the command line)
learning_rate = 0.01
training_epochs = 1
batch_size = 100
num_epochs_record_cost = 1
num_sampled = 5
num_repeat = 1
cost_name =  #
dataset_name = 'wiki10'  # 'AmazonCat'  # 'Bibtex' # 'Eurlex'  # 'mnist'  # 'Delicious' #


results = run(dataset_name, learning_rate, batch_size, num_epochs_record_cost, num_repeat,
              training_epochs, error, cost_name, num_sampled)
pickle.dump(results, open('./Results/' + cost_name + '_' + dataset_name + '.p', 'wb'))

t1 = time.time()
print('Time', t1 - t0)
