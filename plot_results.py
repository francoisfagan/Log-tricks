""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

dataset_name = 'wikiSmall' # 'AmazonCat' # 'Bibtex'  #'wiki10'  #'mnist' #'Delicious' #'Eurlex' #

# Indicate whether to plot train and/or test results
disp_train = True
disp_test = True

# Variations that may iterate over
sgd_names = ['Implicit', 'sampled_softmax', 'Umax', 'nce', 'ove']
initial_learning_rates = [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]

# Default iterate values
initial_learning_rate = 0.01
sgd_name = 'Umax'

# Indicate whether to iterate on sgd or learning rate
iterate_SGD_True_LR_False = True
iterate_list = sgd_names if iterate_SGD_True_LR_False else initial_learning_rates

# Set color spectrum of lines
cmap = plt.get_cmap('jet_r')

# Indicate to use custom optimal learning rate for Eurlex for each algorithm
custom_learning_rate = True

# Plot both classification error and log-loss
for error0_loss1 in [0, 1]:

    # Calculate file name for saving pdf
    pdf_file_name = ('./Results/Plots/'
                     + dataset_name + '_'
                     + ((str(initial_learning_rate) if iterate_SGD_True_LR_False else sgd_name)
                        if not custom_learning_rate else 'custom_lr')
                     + '_'
                     + ('loss' if error0_loss1 else 'error')
                     + '.pdf')

    # Open figure to be saved as pdf
    with PdfPages(pdf_file_name) as pdf:
        plt.figure(figsize=(6, 4))

        # Iterate over variations
        # Record number of iterates that don't open a file
        # This is important to have coloring over the full spectrum
        num_iterates_do_not_open = 0
        for i, iterate in enumerate(iterate_list):

            # Determine variation depending on whether iterating on sgd or learning rate
            if iterate_SGD_True_LR_False:
                sgd_name = iterate
            else:
                initial_learning_rate = iterate

            # Set learning rate if doing sgd
            # Depending on how the file was save, sometimes it should be 100 and other times 100.0
            if custom_learning_rate and iterate_SGD_True_LR_False:
                if sgd_name == 'Implicit':
                    initial_learning_rate = 1.0
                elif sgd_name == 'Umax':
                    initial_learning_rate = 0.1
                elif sgd_name == 'sampled_softmax':
                    initial_learning_rate = 100
                elif sgd_name == 'nce':
                    initial_learning_rate = 100
                elif sgd_name == 'ove':
                    initial_learning_rate = 0.1

            # Set color
            color = cmap(float(i - num_iterates_do_not_open) / (len(iterate_list) - num_iterates_do_not_open))

            # Open the file (if it exists)
            file_name = './Results/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
            if Path(file_name).is_file():
                with open(file_name, 'rb') as f:

                    # Gather the results
                    results = pickle.load(f)
                    epochs = results['epochs'][0]
                    label = str(iterate)

                    # Display mean and standard deviation
                    displays = (['train'] if disp_train else []) + (['test'] if disp_train else [])
                    for display in displays:
                        train_mean = np.mean(results[display][:, :, error0_loss1], axis=0)
                        train_std = np.std(np.array(results[display][:, :, error0_loss1]), axis=0)
                        p = plt.errorbar(epochs,
                                         train_mean,
                                         yerr=train_std,
                                         label=label + ' ' + display,
                                         ecolor=color,
                                         color=color,
                                         linestyle=('dashed' if display == 'test' else 'solid'))

            else:
                num_iterates_do_not_open += 1

        # Create title, labels and legend
        if error0_loss1:
            plt.title(dataset_name + ' loss')
            plt.ylabel('Loss')
        else:
            plt.title(dataset_name + ' error')
            plt.ylabel('Error rate')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')

        # Save figure
        pdf.savefig()
