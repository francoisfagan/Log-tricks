""" Plot results from SGD algorithms


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

dataset_name = 'Eurlex' #'AmazonCat' #'wiki10'  #'mnist' #'Delicious'  #'wikiSmall' #'Bibtex'  #
sgd_name = 'single_nce'#'ove'#'nce' #'sampled_softmax'#'Umax'#'tilde_Umax'#'Implicit'#

# Indicate whether to plot train and/or test results
disp_train = True
disp_test = False

# Variations that may iterate over
sgd_names = ['ove', 'nce', 'sampled_softmax', 'Umax', 'Implicit']  # , 'tilde_Umax'
initial_learning_rates = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]  # 1000000.0,

# Default iterate values
initial_learning_rate = 0.01
# sgd_name = 'tilde_Umax'

# Indicate whether to iterate on sgd or learning rate
iterate_SGD_True_LR_False = False
iterate_list = sgd_names if iterate_SGD_True_LR_False else initial_learning_rates

# Set color spectrum of lines
cmap = plt.get_cmap('jet_r')
linewidth = 0.5
figsize=(3, 3)  # Was (3, 3) for the plots in the main part of the paper

# Indicate to use custom optimal learning rate for Eurlex for each algorithm
custom_learning_rate = False

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
        plt.figure(figsize=figsize)

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

            # Set color
            color = cmap(float(i) / len(iterate_list))

            # Set learning rate if doing sgd
            # Depending on how the file was save, sometimes it should be 100 and other times 100.0
            if custom_learning_rate and iterate_SGD_True_LR_False:
                if sgd_name == 'Implicit':
                    initial_learning_rate = 1.0
                    marker = 'o'
                    color = 'r'
                elif sgd_name == 'Umax':
                    initial_learning_rate = 0.1
                    marker = 'v'
                    color = 'b'
                elif sgd_name == 'sampled_softmax':
                    initial_learning_rate = 100.0
                    marker = 'x'
                    color = 'orange'
                elif sgd_name == 'nce':
                    initial_learning_rate = 100.0
                    marker = '+'
                    color = 'c'
                elif sgd_name == 'ove':
                    initial_learning_rate = 0.1
                    marker = 's'
                    color = 'k'
            else:
                marker = '.'



            # Open the file (if it exists)
            file_name = './Results/Complete/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
            if Path(file_name).is_file():
                with open(file_name, 'rb') as f:
                    print(file_name)

                    # Gather the results
                    results = pickle.load(f)
                    epochs = results['epochs'][0][::2]
                    label = str(iterate)
                    if iterate_SGD_True_LR_False:
                        if iterate == 'sampled_softmax':
                            label = 'IS'
                        elif iterate == 'Umax':
                            label = 'U-max'
                        elif iterate == 'nce':
                            label = 'NCE'
                        elif iterate == 'ove':
                            label = 'OVE'
                        elif iterate == 'single_nce':
                            label = 'NCE (TF)'

                    # Display mean and standard deviation
                    displays = (['train'] if disp_train else []) + (['test'] if disp_test else [])
                    for display in displays:
                        train_mean = np.mean(results[display][:, :, error0_loss1], axis=0)[::2]
                        train_std = np.std(np.array(results[display][:, :, error0_loss1]), axis=0)[::2]
                        p = plt.errorbar(epochs,
                                         train_mean,
                                         yerr=train_std,
                                         label=label,
                                         ecolor=color,
                                         color=color,
                                         linestyle=('dashed' if display == 'test' else 'solid'),
                                         marker=marker,
                                         linewidth=linewidth,
                                         markerfacecolor='none'
                                         )

            else:
                num_iterates_do_not_open += 1

        # Create title, labels and legend
        # if error0_loss1:
        #     plt.ylabel('Log-likelihood')
        # else:
        #     plt.ylabel('Error rate')
        if iterate_SGD_True_LR_False:
            if dataset_name == 'mnist':
                plt.title('MNIST')
            elif dataset_name == 'wiki10':
                plt.title('Wiki10')
            elif dataset_name == 'wikiSmall':
                plt.title('Wiki-small')
            else:
                plt.title(dataset_name)
        else:
            plt.title(sgd_name)
            if sgd_name == 'sampled_softmax':
                plt.title('IS')
            elif sgd_name == 'Umax':
                plt.title('U-max')
            elif sgd_name == 'tilde_Umax':
                plt.title('U-max (2)')
            elif sgd_name == 'nce':
                plt.title('NCE')
            elif sgd_name == 'ove':
                plt.title('OVE')
            elif sgd_name == 'single_nce':
                plt.title('NCE (1,1)')

        # plt.xlabel('Epochs')
        # plt.legend(loc='upper right')

        plt.tight_layout(pad=0.3)

        # Save figure
        pdf.savefig()
