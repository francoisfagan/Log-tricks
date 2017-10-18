"""SGD is run from this file"""

from mnl import *


def run(train, test, learning_rate, num_epochs_record, num_repeat, epochs, sgd_name, num_sampled):

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

        if sgd_name == 'LogTricks':
            sgd = LogTricks(dim, num_classes, num_train_points)
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
                sgd.update(batch_xs, batch_ys, batch_idx, sampled_classes, learning_rate)

            # Record and display loss once each epoch
            if (epoch + 1) % (epochs // num_epochs_record) == 0:

                # Record errors
                train_error[-1].append(sgd.error(train))
                test_error[-1].append(sgd.error(test))
                epochs_recorded[-1].append(epoch)

                print('Epoch:', '%04d' % (epoch + 1),
                      ' Train error:', train_error[-1][-1],
                      ' Test error:', test_error[-1][-1],
                      )

        print('Optimization Finished!')

    record = {'test_error: ': test_error,
              'train_error: ': train_error,
              'epochs_recorded': epochs_recorded
              }

    return record
