"""SGD is run from this file"""

import numpy as np
from load_data import load_data
from mnl import *
from tf_load_data import load_data
from tf_mnl import *


def run(dataset_name,
        initial_learning_rate,
        learning_rate_epoch_decrease,
        num_epochs_record,
        num_repeat,
        epochs,
        sgd_name,
        tf_indicator,
        num_sampled,
        batch_size):

    if tf_indicator:
        train, test, dim, num_classes, num_train_points = load_data(dataset_name)
        variables = graph(dim, num_classes, num_train_points, num_sampled)
        x, y, y_one_hot, W, b = variables
        cost = get_cost(sgd_name, num_classes, num_sampled, *variables)

        optimizer = tf.train.GradientDescentOptimizer(initial_learning_rate).minimize(cost)

    else:
        # Load data
        train, test = load_data(dataset_name)

        # Dataset parameters
        num_train_points = len(train.x)
        dim = train.x[0][0].shape[1]
        num_classes = int(max(train.y)) + 1

    # Error recording data structures
    train_error = []  # Dimension: [num_repeat] x [num_epochs_record]
    test_error = []  # Dimension: [num_repeat] x [num_epochs_record]
    epochs_recorded = []  # Dimension: [num_repeat] x [num_epochs_record]

    for repeat in range(num_repeat):
        print('\nRepetition: ', repeat)

        if sgd_name == 'Umax':
            sgd = Umax(dim, num_classes, num_train_points)
        elif sgd_name == 'Softmax':
            sgd = Softmax(dim, num_classes, num_train_points)
        elif sgd_name == 'Implicit':
            sgd = Implicit(dim, num_classes, num_train_points)
        else:
            print('Please enter a valid sgd method')

        # Prepare new records
        train_error.append([])
        test_error.append([])
        epochs_recorded.append([])

        # Start training
        print('Optimization started!')
        num_batches = num_train_points * (num_sampled if sgd_name == 'Implicit' else 1)
        for epoch in range(epochs):

            # Loop over all batches
            for i_batch in range(num_batches):
                # Get next batch
                batch_xs, batch_ys, batch_idx = train.next_batch(1)
                sampled_classes = np.random.choice(num_classes,
                                                   size=(1 if sgd_name == 'Implicit' else num_sampled),
                                                   replace=False)

                # Take sgd step
                sgd.update(batch_xs,
                           batch_ys,
                           batch_idx,
                           sampled_classes,
                           initial_learning_rate * (learning_rate_epoch_decrease ** epoch)
                           )

            # Record and display loss once each epoch
            if (epoch + 1) % (epochs // num_epochs_record) == 0:
                train_error[-1].append(sgd.error(train))
                test_error[-1].append(sgd.error(test))
                epochs_recorded[-1].append(epoch)

                print('Epoch:', '%04d' % (epoch + 1),
                      ' Train error:', train_error[-1][-1],
                      ' Test error:', test_error[-1][-1],
                      )

        print('Optimization Finished!')

    # Save results
    record = {'test': np.array(test_error),
              'train': np.array(train_error),
              'epochs': np.array(epochs_recorded)
              }

    return record
