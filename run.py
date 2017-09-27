"""SGD is run from this file"""

import tensorflow as tf
import numpy as np
from mnl import u_lower_bound
from tensorflow.python.ops import nn_ops, embedding_ops
from tensorflow.python.ops.nn_impl import _compute_sampled_logits, _sum_rows, sigmoid_cross_entropy_with_logits


def one_hot(y, num_classes):
    return np.eye(num_classes)[y[:, 0]]


# def sample_classes(num_classes, num_sampled, batch_size, batch_ys):
#     # Samples classes excluding the true label
#     # First sample random classes from the set [0, num_classes - 1]
#     # then shift them by 1 + true_label to the set [1 + true_label, num_classes + true_label]
#     # and finally take mod num_classes to the set [0, true_label - 1] union [true_label, num_classes]
#     samples = np.random.randint(num_classes - 1, size=(batch_size, num_sampled))
#     repeated_batch_ys = np.tile(batch_ys, (1, num_sampled))
#     samples = np.mod(samples + repeated_batch_ys + 1, num_classes)
#     return samples


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
        num_sampled,
        x, y, y_one_hot, W, b, idx, u, s_c):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    u_clip = u.assign(tf.maximum(0., u))
    # u_assign_lower_bound = u.assign(tf.maximum(u_lower_bound(x, y, W, b, s_c), u))
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
                    sampled_classes = np.random.randint(num_classes, size=num_sampled)

                    # sess.run(tf.scatter_update(u, tf.squeeze(idx), tf.squeeze(u_lower_bound(x, y, W, b, idx, u, s_c))),
                    #          feed_dict={x: batch_xs,
                    #                     y: batch_ys,
                    #                     idx: batch_idx,
                    #                     s_c: sampled_classes}
                    #          )

                    # Run optimization op (backprop) and cost op (to get loss value)
                    if cost_name != 'softmax':
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y: batch_ys,
                                                   idx: batch_idx,
                                                   s_c: sampled_classes}
                                        )
                    else:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: batch_xs,
                                                   y_one_hot: one_hot(batch_ys, num_classes),
                                                   idx: batch_idx})

                    # Average loss over the batch
                    avg_cost += np.mean(c) / num_batches

                    # Keep u positive
                    sess.run(u_clip)

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
