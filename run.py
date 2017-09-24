"""SGD is run from this file"""


import tensorflow as tf
import numpy as np

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def run(train, test, num_train_points, x, y, W, b, idx, u, cost,
        learning_rate, batch_size, num_epochs_record_cost, num_repeat, training_epochs, error):
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
                    # batch_ys = one_hot(batch_ys, num_classes)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, idx: batch_idx})

                    # Average loss over the batch
                    avg_cost += np.mean(c) / num_batches

                    # Keep u positive
                    sess.run(clip_u)

                # Display logs per epoch step
                if (epoch + 1) % (training_epochs // num_epochs_record_cost) == 0:
                    train_error[-1].append(error(x, y, W, b, train))
                    test_error[-1].append(error(x, y, W, b, test))
                    epochs_recorded[-1].append(epoch)
                    print('Epoch:', '%04d' % (epoch + 1),
                          ' Test error:', test_error[-1][-1],
                          ' Train error:', train_error[-1][-1],)

            print('Optimization Finished!')

    # print(u.eval())


    record = {'test_error: ': test_error,
             'train_error: ': train_error,
             'epochs_recorded': epochs_recorded
             }
    return record