import pickle
import numpy as np
from pathlib import Path

dataset_name = 'AmazonCat' #'wiki10'  #'mnist' #'Delicious'  #'wikiSmall' #'Bibtex'  #'Eurlex' #

# Variations that may iterate over
sgd_names = ['ove', 'nce', 'sampled_softmax', 'Umax', 'Implicit']  # , 'tilde_Umax'

# Plot both classification error and log-loss
for error0_loss1 in [0, 1]:
    if error0_loss1== 0:
        print('\nError')
    else:
        print('\nLog-loss')
    for sgd_name in sgd_names:
        # Determine variation depending on whether iterating on sgd or learning rate

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

        # Open the file (if it exists)
        file_name = './Results/Complete/' + sgd_name + '_' + dataset_name + '_' + str(initial_learning_rate) + '.p'
        if Path(file_name).is_file():
            with open(file_name, 'rb') as f:

                # Gather the results
                results = pickle.load(f)

                # Display mean and standard deviation
                train_mean = np.mean(results['train'][:, :, error0_loss1], axis=0)[-1]
                print(train_mean)