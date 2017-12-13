"""SGD is run from this file"""

import numpy as np

from mnl import *
from tf_mnl import *
import tensorflow as tf
import numpy as np
from tf_load_data import load_data
from tf_mnl import *
import tensorflow as tf

import tf_load_data
import load_data


def run(dataset_name,
        initial_learning_rate,
        learning_rate_epoch_decrease,
        num_epochs_record,
        num_repeat,
        epochs,
        sgd_name,
        tf_indicator,
        num_sampled,
        batch_size,
        proportion_data):
    if tf_indicator:

        train, test, dim, num_classes, num_train_points = tf_load_data.load_data(dataset_name)
        variables = graph(dim, num_classes, num_train_points, num_sampled)
        x, y, y_one_hot, W, b, learning_rate = variables
        cost = get_cost(sgd_name, num_classes, num_sampled, *variables)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    else:

        # Load data
        train, test = load_data.load_data(dataset_name)

        # Dataset parameters
        num_train_points = len(train.x)
        dim = train.x[0][0].shape[1]
        num_classes = int(max(train.y)) + 1

        batch_size = 1  # Only batch_size = 1 is supported

    # Set the number of batches per epoch
    num_batches = num_train_points // batch_size
    # Since 'Implicit' has a sample size of 1, have more batches if 'Implicit'
    # Number dot products if not implicit divided by if implicit = (num_sampled + 1) / 2
    num_batches = int(num_batches * ((num_sampled + 1) / 2 if sgd_name == 'Implicit' else 1.0))
    # If only using a fraction of the data, decrease the num_batches per epoch
    num_batches = int(num_batches * proportion_data)

    # Error recording data structures
    train_error = []  # Dimension: [num_repeat] x [num_epochs_record]
    test_error = []  # Dimension: [num_repeat] x [num_epochs_record]
    epochs_recorded = []  # Dimension: [num_repeat] x [num_epochs_record]

    for repeat in range(num_repeat):
        print('\nRepetition: ', repeat)
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            print('Initializing')
            sess.run(init)

            if sgd_name == 'VanillaSGD':
                sgd = VanillaSGD(dim, num_classes, num_train_points)
            elif sgd_name == 'Umax':
                sgd = Umax(dim, num_classes, num_train_points)
            elif sgd_name == 'tilde_Umax':
                sgd = tilde_Umax(dim, num_classes, num_train_points)
            elif sgd_name == 'Softmax':
                sgd = Softmax(dim, num_classes, num_train_points)
            elif sgd_name == 'Implicit':
                sgd = Implicit(dim, num_classes, num_train_points)

            # Prepare new records
            train_error.append([])
            test_error.append([])
            epochs_recorded.append([])

            # Start training
            print('Optimization started!')

            for epoch in range(epochs):

                # Loop over all batches
                for i_batch in range(num_batches):

                    if tf_indicator:
                        # Get next batch
                        batch_xs, batch_ys = train.next_batch(batch_size, proportion_data)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        if sgd_name != 'softmax':
                            _, c = sess.run([optimizer, cost],
                                            feed_dict={x: batch_xs,
                                                       y: batch_ys,
                                                       learning_rate: initial_learning_rate * (
                                                           learning_rate_epoch_decrease ** epoch)
                                                       }
                                            )
                        else:
                            _, c = sess.run([optimizer, cost],
                                            feed_dict={x: batch_xs,
                                                       y_one_hot: one_hot(batch_ys, num_classes)
                                                       })

                    else:

                        # Get next batch
                        batch_xs, batch_ys, batch_idx = train.next_batch(batch_size, proportion_data)
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
                    if tf_indicator:
                        train_error[-1].append(error(x, y_one_hot, W, b, train, num_classes))
                        test_error[-1].append(error(x, y_one_hot, W, b, test, num_classes))
                        epochs_recorded[-1].append(epoch)
                    else:
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
