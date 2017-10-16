""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'Bibtex'
disp_train = True
disp_test = True

for sgd_name in ['softmax']:
    results = pickle.load(open('Results/' + sgd_name + '_' + dataset_name + '.p', 'rb'))
    epochs = results['epochs_recorded'][0]
    if disp_train:
        train_mean = np.mean(results['train_error: '], axis=0)
        train_std = np.std(results['train_error: '], axis=0)
        plt.errorbar(epochs, train_mean, yerr=train_std, label=sgd_name + ' train')
    if disp_test:
        test_mean = np.mean(results['test_error: '], axis=0)
        test_std = np.std(results['train_error: '], axis=0)
        plt.errorbar(epochs, test_mean, yerr=test_std, linestyle='dashed', label=sgd_name + ' test')
plt.title('Error')
plt.ylabel('Error rate')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()