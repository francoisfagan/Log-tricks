""" Pipeline for softmax optimization experiments

Author: Francois Fagan
"""

from run import run
import time
import sys
import pickle

# Start timing
t0 = time.time()

# Parameters (in the future can pass these in from the command line)
initial_learning_rate = 0.01
learning_rate_epoch_decrease = 0.9
epochs = 50
num_epochs_record = 20
batch_size = 100  # For numpy methods batch_size = 1 always
num_sampled = 5
num_repeat = 1
sgd_name = 'nce'
dataset_name = 'Bibtex' # 'Eurlex' #'wikiSmall'
custom_learning_rate = False  # Indicate to use custom optimal learning rate for Eurlex for each algorithm

# Read in parameters if passed in from the command line
if len(sys.argv) > 1:
    sgd_name = sys.argv[1]
    dataset_name = sys.argv[2]
    initial_learning_rate = 10 ** float(sys.argv[3])

# Set custom learning rate for given algorithm
if custom_learning_rate:
    if sgd_name == 'Implicit':
        initial_learning_rate = 1.0
    elif sgd_name == 'Umax':
        initial_learning_rate = 0.1
    elif sgd_name == 'tilde_Umax':
        initial_learning_rate = 0.1
    elif sgd_name == 'sampled_softmax':
        initial_learning_rate = 100
    elif sgd_name == 'nce':
        initial_learning_rate = 100
    elif sgd_name == 'ove':
        initial_learning_rate = 0.1

print('initial_learning_rate', initial_learning_rate)

# Select algorithm and dataset
# for sgd_name in ['sampled_softmax']:
#     # 'Implicit', 'Umax', 'sampled_softmax', 'nce', 'ove', ### 'Softmax', 'tf_softmax'
#     for dataset_name in ['Bibtex']:
# 'wikiSmall'  # 'AmazonCat'  # 'wiki10'  # 'Eurlex'  # 'mnist'  # 'Delicious'
print(sgd_name, dataset_name)
tf_indicator = sgd_name in {'sampled_softmax', 'tf_softmax', 'nce', 'ove'}

# Run the algorithm!
record = run(dataset_name,
             initial_learning_rate,
             learning_rate_epoch_decrease,
             num_epochs_record,
             num_repeat,
             epochs,
             sgd_name,
             tf_indicator,
             num_sampled,
             batch_size)

file_name = './Results/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
pickle.dump(record, open(file_name, 'wb'))

# Print how long the program took
t1 = time.time()
print('Time', t1 - t0)
