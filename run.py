"""SGD is run from this file"""

import tensorflow as tf
import numpy as np


def one_hot(y, num_classes):
    return np.eye(num_classes)[y[:, 0]]


def measure_u(train, num_classes, W, b, u):
    # Measures difference between exp(u_i) and 1 + \sum_{k\neq y_i}\exp(x_i^\top(w_k-w_{y_i}))
    print('Started measuring u')
    difference = 0
    W_np = W.eval()
    b_np = b.eval()
    u_np = u.eval()
    for i in range(train.x.shape[0]):
        y_i = train.y[i][0]
        y_i_one_hot = np.eye(int(num_classes))[y_i]
        x_i = train.x[i, :]
        true_inner = np.dot(x_i, W_np[:, y_i]) + b_np[y_i]
        denominator_i = np.exp(u_np[i])
        # denominator_i = np.dot(x_i, W_np[:, y_i]) + b_np[y_i]
        # denominator_i = np.max(np.dot(x_i, W_np) + b_np - true_inner)
        # denominator_i = (1
        #                  + np.exp(-(np.dot(x_i, W_np[:, y_i]) + b_np[y_i]))
        #                  * np.dot(1 - y_i_one_hot, np.exp(np.dot(x_i, W_np) + b_np)))
        difference_i = denominator_i  # abs(np.exp(u_np[i]) - denominator_i)
        difference += difference_i
    print('difference:', difference / train.x.shape[0])
    return difference


def run(train, test, num_train_points, cost,
        learning_rate, batch_size, num_epochs_record_cost, num_repeat, training_epochs, error, num_classes, cost_name,
        x, y, y_one_hot, W, b, idx, u):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    clip_u = u.assign(tf.maximum(0., u))
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
                    batch_xs, batch_ys, batch_idx = train.next_batch(batch_size)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    if cost_name == 'softmax':
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y_one_hot: one_hot(batch_ys, num_classes),
                                                   idx: batch_idx})
                    else:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y: batch_ys,
                                                   idx: batch_idx})

                    # Average loss over the batch
                    avg_cost += np.mean(c) / num_batches

                    # Keep u positive
                    sess.run(clip_u)

                # Display logs per epoch step
                if (epoch + 1) % (training_epochs // num_epochs_record_cost) == 0:
                    measure_u(train, num_classes, W, b, u)
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
