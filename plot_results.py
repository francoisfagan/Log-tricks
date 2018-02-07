""" Plot results from SGD algorithms


"""

import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

dataset_name = 'wiki10'  #'AmazonCat' #'Eurlex'  #'Delicious'  #'Bibtex'  #'mnist' #'wikiSmall'  #
sgd_name = 'Umax'  # 'tilde_Umax'#'VanillaSGD' #'sampled_softmax'  # 'nce' #'ove' #'Implicit' #
# 'single_nce'#

# Variations that may iterate over
sgd_names = ['ove', 'nce', 'sampled_softmax', 'VanillaSGD', 'Umax', 'Implicit']  # , 'tilde_Umax'
initial_learning_rates = [0.001, 0.0001]  # 1000000.0,, 0.00001,, 0.0001, 1000.0, 100.0, 10.0, 1.0, 0.1,

# Default iterate values
initial_learning_rate = 0.01
# sgd_name = 'tilde_Umax'

# Indicate whether to plot train and/or test results
disp_train = True
disp_test = False

# Indicate whether to iterate on sgd or learning rate
iterate_SGD_True_LR_False = True
iterate_list = sgd_names if iterate_SGD_True_LR_False else initial_learning_rates

# Set color spectrum of lines
cmap = plt.get_cmap('jet_r')
linewidth = 0.5
figsize = (8, 8)  # Was (3, 3) for the plots in the main part of the paper

# Plot both classification error and log-loss
for error0_loss1 in [0, 1]:
    print('loss' if error0_loss1 else 'error')

    # Calculate file name for saving pdf
    pdf_file_name = ('./Results/Plots/'
                     + dataset_name + '_'
                     + ((str(initial_learning_rate) if iterate_SGD_True_LR_False else sgd_name)
            if not iterate_SGD_True_LR_False else 'custom_lr')
                     + '_'
                     + ('loss' if error0_loss1 else 'error')
                     + '.pdf')

    # Open figure to be saved as pdf
    with PdfPages(pdf_file_name) as pdf:
        plt.figure(figsize=figsize)

        # for sgd_name in ['Umax', 'VanillaSGD']:  #  Use for double-sum plots

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
            color_shift = 5
            color = cmap((float(i) + color_shift) / (len(iterate_list) + color_shift))

            # Set learning rate if doing sgd
            # Depending on how the file was save, sometimes it should be 100 and other times 100.0
            if iterate_SGD_True_LR_False:
                if sgd_name == 'Implicit':
                    initial_learning_rate = 1.0
                    marker = 'o'
                    color = 'r'
                elif sgd_name == 'Umax':
                    initial_learning_rate = 0.001
                    marker = 'v'
                    color = 'b'
                elif sgd_name == 'VanillaSGD':
                    initial_learning_rate = 0.0001
                    marker = 'd'
                    color = 'm'
                elif sgd_name == 'sampled_softmax':
                    initial_learning_rate = 100.0
                    marker = 'x'
                    color = 'orange'
                elif sgd_name == 'nce':
                    initial_learning_rate = 1000.0
                    marker = '+'
                    color = 'c'
                elif sgd_name == 'ove':
                    initial_learning_rate = 1000.0
                    marker = 's'
                    color = 'k'
            else:
                marker = '.'

            # Set path and file prefix
            if iterate_SGD_True_LR_False:
                path = './Results/Complete/Tuned/'
                file_name_prefix = sgd_name + '_' + dataset_name
            else:
                path = './Results/Complete/Tuning/'
                file_name_prefix = sgd_name + '_' + dataset_name + '_lr_' + str(initial_learning_rate)

            if dataset_name == 'Eurlex' and sgd_name in {'Umax', 'tilde_Umax'}:
                path = './Results/Complete/Original/'
                file_name_prefix = sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate)

            # Extract the full file name
            matching_file_names = [filename for filename in os.listdir(path) if
                                   filename.startswith(file_name_prefix)]
            if not matching_file_names:
                print('No matching filenames')
                continue  # There are no matching filenames
            file_name = path + matching_file_names[0]

            # print(file_name)

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
                        print(train_mean)


                        if label == 'Implicit':
                            p = plt.errorbar(epochs[::2],
                                             train_mean[:len(epochs[::2])],
                                             yerr=train_std[:len(epochs[::2])],
                                             label='Implicit (lagged)',
                                             ecolor=color,
                                             color=color,
                                             linestyle='solid',
                                             marker=marker,
                                             linewidth=linewidth,
                                             )

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


                        # Use code below for double-sum plots
                        # # color = 'k'
                        # # label = ('Raman et al.' if sgd_name == 'tilde_Umax' else 'Proposed')
                        # p = plt.plot(epochs,  #
                        #              train_mean,
                        #              # yerr=train_std,
                        #              label=sgd_name + ' ' + str(initial_learning_rate),  # label,
                        #              # ecolor=color,
                        #              color=color,
                        #              # linestyle=('solid' if initial_learning_rate == 0.001 else 'dashed'),
                        #              marker=('o' if sgd_name == 'Umax' else 'd'),
                        #              # linewidth=linewidth,
                        #              markerfacecolor=(color if sgd_name == 'Umax' else 'none')
                        #              )


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
                plt.title('U-max (Raman)')
            elif sgd_name == 'nce':
                plt.title('NCE')
            elif sgd_name == 'ove':
                plt.title('OVE')
            elif sgd_name == 'single_nce':
                plt.title('NCE (1,1)')

        # plt.title('Double-sum formulations')
        # plt.title('Umax vs VanillaSGD')

        # plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        plt.legend(loc=3, bbox_to_anchor=(1.05, 0.63))

        plt.tight_layout(pad=0.3)

        # Save figure
        pdf.savefig(bbox_inches="tight")
