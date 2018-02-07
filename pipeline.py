""" Pipeline for softmax optimization experiments

Author: Francois Fagan
"""

from run import run
import sys
import pickle

# Parameters (in the future can pass these in from the command line)
initial_learning_rate = 0.1
learning_rate_epoch_decrease = 0.9
epochs = 1
num_epochs_record = 1
batch_size = 100  # For numpy methods batch_size = 1 always
proportion_data = 1  # If =1 the use all of the data, else use a subset. For parameter tuning purposes.
num_sampled = 5
num_repeat = 1
sgd_name = 'Implicit' #'VanillaSGD' #_simple
dataset_name = 'mnist'
custom_learning_rate = False  # Indicate to use custom optimal learning rate for Eurlex for each algorithm

# Read in parameters if passed in from the command line
if len(sys.argv) > 1:
    sgd_name = sys.argv[1]
    dataset_name = sys.argv[2]
    initial_learning_rate = 10 ** float(sys.argv[3])
    proportion_data = float(sys.argv[4])

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
# 'wikiSmall'  # 'wiki10'  # 'Eurlex'  # 'mnist'  # 'Delicious'
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
             batch_size,
             proportion_data)

file_name = './Results/' + sgd_name + '_' + dataset_name + '_lr_' + str(initial_learning_rate) + '_prop_data_' + str(proportion_data) + '.p'
pickle.dump(record, open(file_name, 'wb'))
