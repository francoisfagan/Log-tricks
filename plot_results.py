""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dataset_name = 'Bibtex'  # 'Delicious' #'Eurlex' #
disp_train = True
disp_test = True
error0_loss1 = 1
assert (error0_loss1 in {0, 1})
initial_learning_rate = 0.0001
sgd_names = ['Implicit']  #, 'Umax', 'Softmax'

cmap = plt.get_cmap('jet_r')  # To get train and test plots the same color for each algorithm
for error0_loss1 in [0, 1]:
    with PdfPages('./Results/Plots/' + dataset_name + '_' + ('loss' if error0_loss1 else 'error') + '.pdf') as pdf:
        plt.figure(figsize=(6, 4))
        for i, sgd_name in enumerate(sgd_names):
            color = cmap(float(i) / len(sgd_names))
            file_name = './Results/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
            results = pickle.load(open(file_name, 'rb'))
            epochs = results['epochs'][0]
            if disp_train:
                train_mean = np.mean(results['train'][:, :, error0_loss1], axis=0)
                train_std = np.std(np.array(results['train'][:, :, error0_loss1]), axis=0)
                p = plt.errorbar(epochs, train_mean, yerr=train_std, label=sgd_name + ' train',
                                 ecolor=color, color=color)
            if disp_test:
                test_mean = np.mean(np.array(results['test'][:, :, error0_loss1]), axis=0)
                test_std = np.std(np.array(results['test'][:, :, error0_loss1]), axis=0)
                plt.errorbar(epochs, test_mean,
                             yerr=test_std, linestyle='dashed', label=sgd_name + ' test',
                             ecolor=color, color=color)

        if error0_loss1:
            plt.title(dataset_name + ' loss')
            plt.ylabel('Loss')
        else:
            plt.title(dataset_name + ' error')
            plt.ylabel('Error rate')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        pdf.savefig()
