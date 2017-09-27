"""Multinomial Regression Loss variables and cost functions are defined here.

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn_impl import _compute_sampled_logits, _sum_rows, sigmoid_cross_entropy_with_logits
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler
from tensorflow.python.ops.gen_nn_ops import softplus
from tensorflow.python.ops import nn_ops, embedding_ops, math_ops, array_ops, variables, candidate_sampling_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


def one_hot(y, num_classes):
    # Takes y of shape [batch_size, 1] and outputs one-hot version of shape [batch_size, num_classes]
    return np.eye(num_classes)[y[:, 0]]


def _compute_sampled_logits_with_samples(weights,
                                         biases,
                                         labels,
                                         inputs,
                                         sampled,
                                         num_true=1,
                                         partition_strategy="mod",
                                         name=None):
    """Helper function for nce_loss and sampled_softmax_loss functions.
    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).
    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.
    Args:
      weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
          objects whose concatenation along dimension 0 has shape
          `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
      biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
          class biases.
      labels: A `Tensor` of type `int64` and shape `[batch_size,
          num_true]`. The target classes.  Note that this format differs from
          the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
          activations of the input network.
      num_sampled: An `int`.  The number of classes to randomly sample per batch.
      num_classes: An `int`. The number of possible classes.
      num_true: An `int`.  The number of target classes per training example.
      sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
          `sampled_expected_count`) returned by a `*_candidate_sampler` function.
          (if None, we default to `log_uniform_candidate_sampler`)
      subtract_log_q: A `bool`.  whether to subtract the log expected count of
          the labels in the sample to get the logits of the true labels.
          Default is True.  Turn off for Negative Sampling.
      remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
          where a sampled class equals one of the target classes.  Default is
          False.
      partition_strategy: A string specifying the partitioning strategy, relevant
          if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
          Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: A name for the operation (optional).
    Returns:
      out_logits, out_labels: `Tensor` objects each with shape
          `[batch_size, num_true + num_sampled]`, for passing to either
          `nn.sigmoid_cross_entropy_with_logits` (NCE) or
          `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """

    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "compute_sampled_logits",
                        weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        # if sampled_values is None:
        #     sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
        #         true_classes=labels,
        #         num_true=num_true,
        #         num_sampled=num_sampled,
        #         unique=True,
        #         range_max=num_classes)
        # # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # # pylint: disable=unpacking-non-sequence
        # sampled, true_expected_count, sampled_expected_count = (
        #     array_ops.stop_gradient(s) for s in sampled_values)
        # # pylint: enable=unpacking-non-sequence
        # sampled = math_ops.cast(sampled, dtypes.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = array_ops.concat([labels_flat, sampled], 0)

        # Retrieve the true weights and the logits of the sampled weights.

        # weights shape is [num_classes, dim]
        all_w = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)

        # true_w shape is [batch_size * num_true, dim]
        true_w = array_ops.slice(all_w, [0, 0],
                                 array_ops.stack(
                                     [array_ops.shape(labels_flat)[0], -1]))

        sampled_w = array_ops.slice(
            all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # Apply X*W', which yields [batch_size, num_sampled]
        sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_b = embedding_ops.embedding_lookup(
            biases, all_ids, partition_strategy=partition_strategy)
        # true_b is a [batch_size * num_true] tensor
        # sampled_b is a [num_sampled] float tensor
        true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
        sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
        row_wise_dots = math_ops.multiply(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat([[-1], dim], 0))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = array_ops.reshape(true_b, [-1, num_true])
        true_logits += true_b
        sampled_logits += sampled_b

        # if remove_accidental_hits:
        #     acc_hits = candidate_sampling_ops.compute_accidental_hits(
        #         labels, sampled, num_true=num_true)
        #     acc_indices, acc_ids, acc_weights = acc_hits
        #
        #     # This is how SparseToDense expects the indices.
        #     acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
        #     acc_ids_2d_int32 = array_ops.reshape(
        #         math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
        #     sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
        #                                       "sparse_indices")
        #     # Create sampled_logits_shape = [batch_size, num_sampled]
        #     sampled_logits_shape = array_ops.concat(
        #         [array_ops.shape(labels)[:1],
        #          array_ops.expand_dims(num_sampled, 0)], 0)
        #     if sampled_logits.dtype != acc_weights.dtype:
        #         acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
        #     sampled_logits += sparse_ops.sparse_to_dense(
        #         sparse_indices,
        #         sampled_logits_shape,
        #         acc_weights,
        #         default_value=0.0,
        #         validate_indices=False)
        #
        # if subtract_log_q:
        #     # Subtract log of Q(l), prior probability that l appears in sampled.
        #     true_logits -= math_ops.log(true_expected_count)
        #     sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
            array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)
        ], 1)

    return out_logits, out_labels


def graph(dim, num_classes, num_train_points, num_sampled):
    print('Defining graph')

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, dim])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.int64, [None, 1])  # 0-9 digits recognition => 10 classes
    y_one_hot = tf.placeholder(tf.float32, [None, num_classes])  # 0-9 digits recognition => 10 classes
    idx = tf.placeholder(tf.int64, [None, 1])  # data point indices
    s_c = tf.placeholder(tf.int64, num_sampled)  # sampled classes

    # Set model weights
    W = tf.Variable(tf.zeros([dim, num_classes]))
    b = tf.zeros([num_classes])
    u = tf.Variable(tf.ones([num_train_points]) * tf.log(float(num_classes)))  # Initialize u_i = log(K)

    variables = [x, y, y_one_hot, W, b, idx, u, s_c]

    return variables


def error(x, y_one_hot, W, b, data, num_classes):
    pred_softmax = tf.nn.softmax(tf.matmul(x, W) + b)
    wrong_prediction = tf.not_equal(tf.argmax(pred_softmax, 1), tf.argmax(y_one_hot, 1))
    # Calculate accuracy
    pred_error = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))
    return pred_error.eval({x: data.x, y_one_hot: one_hot(data.y, num_classes)})


def get_cost(cost_name, num_classes, num_sampled, x, y, y_one_hot, W, b, idx, u, s_c):
    print('Getting cost function')

    if cost_name == 'softmax':
        return cost_softmax(x, y_one_hot, W, b)
    elif cost_name == 'sampled_softmax':
        return cost_sampled_softmax(x, y, W, b, num_classes, num_sampled)
    elif cost_name == 'nce':
        return cost_nce(x, y, W, b, num_classes, num_sampled)
    elif cost_name == 'ove':
        return cost_ove(x, y, W, b, num_classes, num_sampled)
    elif cost_name == 'lt':
        return cost_lt(x, y, W, b, idx, u, s_c, num_classes, num_sampled)


def cost_softmax(x, y_one_hot, W, b):
    # Softmax without sampling
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y_one_hot * tf.log(pred), reduction_indices=1))
    return cost


def sampled_cost_wrapper(cost_function, sampler=uniform_candidate_sampler, subtract_log_q=True):
    # Decorator for sampled cost functions
    # This allows one to easily define new sampled cost functions
    # It does the sampling of the indices and then extracts the corresponding logits and labels
    # The logits and labels can then be fed into any given cost_function
    # It uses the _compute_sampled_logits function to sample random classes, available here:
    # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py

    def cost(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes
             ):
        sampled_values = sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes,
        )

        logits, labels = _compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
            num_sampled=num_sampled,
            num_classes=num_classes,
            num_true=1,
            sampled_values=sampled_values,
            subtract_log_q=subtract_log_q
        )

        return cost_function(labels=labels, logits=logits, num_classes=num_classes)

    return cost


def cost_sampled_softmax(x, y, W, b, num_classes, num_sampled):
    # Sampled softmax = Importance sampling

    # If we wanted to sample the points using the log-uniform (Zipfian) base distribution
    # then we could just return:
    # return tf.nn.sampled_softmax_loss(weights=tf.transpose(W),
    #                                   biases=b,
    #                                   inputs=x,
    #                                   labels=y,
    #                                   num_sampled=num_sampled,
    #                                   num_classes=num_classes)
    #
    # However we want to sample the points uniformly.
    # It is easiest to do this using the sampled_cost_wrapper.
    # The cost function for nce is taken directly from
    # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py

    def cost_function(labels, logits, num_classes):
        sampled_losses = nn_ops.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return sampled_losses

    cost = sampled_cost_wrapper(cost_function, sampler=uniform_candidate_sampler, subtract_log_q=True)

    return cost(weights=tf.transpose(W),
                biases=b,
                labels=y,
                inputs=x,
                num_sampled=num_sampled,
                num_classes=num_classes
                )


def cost_nce(x, y, W, b, num_classes, num_sampled):
    # Noise Contrastive Estimation

    # If we wanted to sample the points using the log-uniform (Zipfian) base distribution
    # then we could just return:
    # return tf.nn.nce_loss(weights=tf.transpose(W),
    #                       biases=b,
    #                       inputs=x,
    #                       labels=y,
    #                       num_sampled=num_sampled,
    #                       num_classes=num_classes)
    #
    # However we want to sample the points uniformly.
    # It is easiest to do this using the sampled_cost_wrapper.
    # The cost function for nce is taken directly from
    # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py

    def cost_function(labels, logits, num_classes):
        sampled_losses = sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, name="sampled_losses")
        return _sum_rows(sampled_losses)

    cost = sampled_cost_wrapper(cost_function, sampler=uniform_candidate_sampler, subtract_log_q=True)

    return cost(weights=tf.transpose(W),
                biases=b,
                labels=y,
                inputs=x,
                num_sampled=num_sampled,
                num_classes=num_classes
                )


def cost_ove(x, y, W, b, num_classes, num_sampled):
    # One-vs-each

    # Michalis Titsias "One-vs-each approximation to softmax for scalable estimation of probabilities."
    # Advances in Neural Information Processing Systems. 2016.

    sgd_weight = float(num_classes) / float(num_sampled)

    def cost_function(labels, logits, num_classes):
        true_logit = _sum_rows(labels * logits)
        repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
        logit_difference = logits - repeated_true_logit
        # Multiply the cost by float(num_classes) to account for the uniform sampling
        return sgd_weight * _sum_rows((1 - labels) * softplus(logit_difference))

    cost = sampled_cost_wrapper(cost_function, sampler=uniform_candidate_sampler, subtract_log_q=False)

    return cost(weights=tf.transpose(W),
                biases=b,
                labels=y,
                inputs=x,
                num_sampled=num_sampled,
                num_classes=num_classes
                )


def cost_lt(x, y, W, b, idx, u, s_c, num_classes, num_sampled):
    # Log-tricks

    sgd_weight = float(num_classes) / float(num_sampled)

    logits, labels = _compute_sampled_logits_with_samples(
        weights=tf.transpose(W),
        biases=b,
        labels=y,
        inputs=x,
        sampled=s_c
    )

    u_idx = tf.transpose(embedding_ops.embedding_lookup(u, idx))
    true_logit = _sum_rows(labels * logits) + u_idx
    repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
    logit_difference = logits - repeated_true_logit
    return u_idx + tf.exp(-u_idx) + sgd_weight * _sum_rows((1 - labels) * tf.exp(logit_difference))


def u_lower_bound(x, y, W, b, idx, u, s_c):
    # Lower bound on u given the samples

    logits, labels = _compute_sampled_logits_with_samples(
        weights=tf.transpose(W),
        biases=b,
        labels=y,
        inputs=x,
        sampled=s_c
    )
    u_idx = tf.transpose(embedding_ops.embedding_lookup(u, idx))
    true_logit = _sum_rows(labels * logits)
    repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
    logit_difference = logits - repeated_true_logit
    return tf.transpose(tf.maximum(u_idx, tf.log(1.0 + _sum_rows((1 - labels) * tf.exp(logit_difference)))))

    #
    # # ------------------------------------
    # # Loss functions
    # # ------------------------------------
    #
    #
    # # Negative sampling without sampling
    # pred_negative_sampling = tf.nn.sigmoid(tf.matmul(x, W) + b)
    # cost_negative_sampling = tf.reduce_mean(-tf.reduce_sum((
    #     y * tf.log(pred_negative_sampling)
    #     + (1 - y) * tf.log(1 - pred_negative_sampling)
    # ),
    #     reduction_indices=1))
    #
    #
    # def debais_cost_fn(W, b, x, y):
    #     pred_softmax = tf.nn.softmax(tf.matmul(x, W) + b)
    #     cost_softmax = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred_softmax),
    #                                                  reduction_indices=1))
    #
    #     def f1(): return 0.0 * cost_softmax
    #
    #     def f2(): return 10.0 * cost_softmax
    #
    #     return tf.cond(tf.less(tf.random_uniform([]), tf.constant(0.999)), f1, f2)
    #
    # debais_cost = debais_cost_fn(W, b, x, y)


class LogTricks:

    def __init__(self, dim, num_classes, num_train_points):
        self.W = np.zeros((dim, num_classes))
        self.u = np.zeros(num_train_points)


    def sgd_update(self, x, y, idx, s_c, learning_rate):
        """

        :param x: np.array of dimensions [batch_size] x [dim]
        :param y: np.array of dimensions [batch_size] x [1]
        :param idx: np.array of dimensions [batch_size] x [1]
        :param s_c:
        :param learning_rate:
        :return:
        """

        labels =
        logits =