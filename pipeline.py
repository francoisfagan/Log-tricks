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
initial_learning_rate = 0.0001
learning_rate_epoch_decrease = 0.9
epochs = 10
num_epochs_record = 10
batch_size = 100
num_sampled = 5
num_repeat = 1

# Read in parameters if passed in from the command line
if len(sys.argv) > 1:
    sgd_name = sys.argv[1]
    dataset_name = sys.argv[2]

# Select algorithm and dataset
for sgd_name in ['Implicit']:
    # , 'Umax', 'Softmax', 'sampled_softmax'  # 'tf_softmax'  # 'nce' # 'ove'
    tf_indicator = sgd_name in {'sampled_softmax', 'tf_softmax', 'nce', 'ove'}
    for dataset_name in ['Bibtex']:
        # 'wikiSmall'  # 'AmazonCat'  # 'wiki10'  # 'Eurlex'  # 'mnist'  # 'Delicious'
        print(sgd_name, dataset_name)

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
