""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dataset_name = 'Delicious' #'Eurlex' #'Bibtex' #
disp_train = True
disp_test = True
sgd_names = ['Umax', 'Implicit', 'Softmax']

cmap = plt.get_cmap('jet_r')
with PdfPages('./Results/Plots/' + dataset_name + '.pdf') as pdf:
    plt.figure(figsize=(6, 4))
    for i, sgd_name in enumerate(sgd_names):
        color = cmap(float(i) / len(sgd_names))
        results = pickle.load(open('Results/' + sgd_name + '_' + dataset_name + '.p', 'rb'))
        epochs = results['epochs_recorded'][0]
        if disp_train:
            train_mean = np.mean(results['train_error: '], axis=0)
            train_std = np.std(results['train_error: '], axis=0)
            p = plt.errorbar(epochs, train_mean, yerr=train_std, label=sgd_name + ' train',
                             ecolor=color, color=color)
        if disp_test:
            test_mean = np.mean(results['test_error: '], axis=0)
            test_std = np.std(results['train_error: '], axis=0)

            # if disp_train:
            #     train_color = p[0].get_color()
            #     print(train_color)
            #     plt.errorbar(epochs, test_mean,
            #                  yerr=test_std, linestyle='dashed', label=sgd_name + ' test',
            #                  ecolor=train_color)
            # else:
            plt.errorbar(epochs, test_mean,
                             yerr=test_std, linestyle='dashed', label=sgd_name + ' test',
                             ecolor=color, color=color)

    plt.title(dataset_name + ' error')
    plt.ylabel('Error rate')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    pdf.savefig()
    # plt.show()