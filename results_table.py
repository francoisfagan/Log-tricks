import pickle
import os
import numpy as np
from pathlib import Path

dataset_name = 'wikiSmall' #'wiki10'  #'AmazonCat'  #'Eurlex' #'Delicious'  #'Bibtex'  #'mnist' #
print(dataset_name)
# Variations that may iterate over
sgd_names = ['Implicit', 'ove', 'nce', 'sampled_softmax', 'VanillaSGD', 'Umax']  # , 'tilde_Umax'

# Plot both classification error and log-loss
for error0_loss1 in [0, 1]:
    if error0_loss1 == 0:
        print('\nError')
    else:
        print('\nLog-loss')

    implicit_value = 1.0

    for sgd_name in sgd_names:

        # Set path and file prefix
        path = './Results/Complete/Tuned/'
        file_name_prefix = sgd_name + '_' + dataset_name

        # Extract the full file name
        matching_file_names = [filename for filename in os.listdir(path) if
                               filename.startswith(file_name_prefix)]
        if not matching_file_names:
            continue  # There are no matching filenames
        file_name = path + matching_file_names[0]

        # Open the file (if it exists)
        # file_name = './Results/Complete/Tuned/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
        if Path(file_name).is_file():
            with open(file_name, 'rb') as f:
                # Gather the results
                results = pickle.load(f)

                # Display mean and standard deviation
                train_mean = np.mean(results['train'][:, :, error0_loss1], axis=0)[-1]

                if sgd_name == 'Implicit':
                    implicit_value = train_mean

                print(sgd_name + ': {:.2f}'.format(train_mean / implicit_value))
