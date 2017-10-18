"""SGD is run from this file"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn_ops, embedding_ops
from tensorflow.python.ops.nn_impl import _compute_sampled_logits, _sum_rows, sigmoid_cross_entropy_with_logits


def one_hot(y, num_classes):
    return np.eye(num_classes)[y[:, 0]]


def run(train, test, num_train_points, cost,
        learning_rate, batch_size, num_epochs_record_cost, num_repeat, training_epochs, error, num_classes, cost_name,
        num_sampled,
        x, y, y_one_hot, W, b):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    train_error = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]
    test_error = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]
    epochs_recorded = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]
    for repeat in range(num_repeat):
        print('\nRepetition: ', repeat)
        train_error.append([])
        test_error.append([])
        epochs_recorded.append([])

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            print('Initializing')
            sess.run(init)

            print('Optimization started!')
            num_batches = int(num_train_points / batch_size)
            for epoch in range(training_epochs):
                avg_cost = 0.

                # Loop over all batches
                for i_batch in range(num_batches):
                    # Get next batch
                    batch_xs, batch_ys = train.next_batch(batch_size)
                    sampled_classes = np.random.randint(num_classes, size=num_sampled)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    if cost_name != 'softmax':
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y: batch_ys
                                                   }
                                        )
                    else:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y_one_hot: one_hot(batch_ys, num_classes)
                                                   })

                    # Average loss over the batch
                    avg_cost += np.mean(c) / num_batches

                # Display logs per epoch step
                if (epoch + 1) % (training_epochs // num_epochs_record_cost) == 0:
                    # measure_u(train, num_classes, W, b, u)
                    train_error[-1].append(error(x, y_one_hot, W, b, train, num_classes))
                    test_error[-1].append(error(x, y_one_hot, W, b, test, num_classes))
                    epochs_recorded[-1].append(epoch)
                    print('Epoch:', '%04d' % (epoch + 1),
                          'ave_cost', avg_cost,
                          ' Test error:', test_error[-1][-1],
                          ' Train error:', train_error[-1][-1], )

            print('Optimization Finished!')

    record = {'test_error: ': test_error,
              'train_error: ': train_error,
              'epochs_recorded': epochs_recorded
              }
    return record
