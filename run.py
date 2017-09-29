"""SGD is run from this file"""

import tensorflow as tf
import numpy as np
from mnl import *
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
    train_error = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]
    test_error = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]
    epochs_recorded = []  # Cost list of lists of dim: [num_repeat] x [num_epochs_record_cost]


    for repeat in range(num_repeat):
        print('\nRepetition: ', repeat)

        if cost_name in {'lt', 'IS', 'softmax_IS'}:
            lt = LogTricks(train.x.shape[1], num_classes, num_train_points)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
                    sampled_classes = np.random.choice(num_classes, size=num_sampled, replace=False)
                    if cost_name == 'softmax_IS':
                        sampled_classes = np.arange(num_classes)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    if cost_name in {'lt', 'IS', 'softmax_IS'}:
                        u_equals_bound = (cost_name in {'IS', 'softmax_IS'})
                        lt.sgd_update(batch_xs, batch_ys, batch_idx, sampled_classes, learning_rate, u_equals_bound)
                    elif cost_name != 'softmax':
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

                # Display logs per epoch step
                if (epoch + 1) % (training_epochs // num_epochs_record_cost) == 0:
                    print('Epoch:', '%04d' % (epoch + 1))
                    if cost_name in {'lt', 'IS', 'softmax_IS'}:
                        epoch_train_error = lt.lt_error(train)
                        epoch_test_error = lt.lt_error(test)
                    else:
                        epoch_train_error = error(x, y_one_hot, W, b, train, num_classes)
                        epoch_test_error = error(x, y_one_hot, W, b, test, num_classes)

                    print(
                        ' Test error:', epoch_test_error,
                        ' Train error:', epoch_train_error,
                    )
                    train_error[-1].append(epoch_train_error)
                    test_error[-1].append(epoch_test_error)
                    epochs_recorded[-1].append(epoch)
                    # measure_u(train, num_classes, W, b, u)

            print('Optimization Finished!')

    record = {'test_error: ': test_error,
              'train_error: ': train_error,
              'epochs_recorded': epochs_recorded
              }
    return record
