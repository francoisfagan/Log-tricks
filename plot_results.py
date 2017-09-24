""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

cost_name = 'softmax'
dataset_name = 'Delicious'

record = pickle.load( open(cost_name + '_' + dataset_name + '.p', 'rb'))
train_mean = np.mean(record['train_error: '], axis=0)
test_mean = np.mean(record['test_error: '], axis=0)
epochs = record['epochs_recorded'][0]
plt.plot(epochs, train_mean, label = 'train')
plt.plot(epochs, test_mean, label = 'test')
plt.title('Error')
plt.legend()
plt.show()