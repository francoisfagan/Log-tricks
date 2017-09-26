"""Multinomial Regression Loss variables and cost functions are defined here.

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn_impl import _compute_sampled_logits, _sum_rows, sigmoid_cross_entropy_with_logits
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler
from tensorflow.python.ops.gen_nn_ops import softplus
from tensorflow.python.ops import nn_ops, embedding_ops


def one_hot(y, num_classes):
    # Takes y of shape [batch_size, 1] and outputs one-hot version of shape [batch_size, num_classes]
    return np.eye(num_classes)[y[:, 0]]


def graph(dim, num_classes, num_train_points):
    print('Defining graph')

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, dim])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.int64, [None, 1])  # 0-9 digits recognition => 10 classes
    y_one_hot = tf.placeholder(tf.float32, [None, num_classes])  # 0-9 digits recognition => 10 classes
    idx = tf.placeholder(tf.int64, [None, 1])  # data point indices

    # Set model weights
    W = tf.Variable(tf.zeros([dim, num_classes]))
    b = tf.zeros([num_classes])  # tf.Variable(
    u = tf.Variable(tf.ones([num_train_points]) * tf.log(float(num_classes)))  # Initialize u_i = log(K)

    variables = [x, y, y_one_hot, W, b, idx, u]

    return variables


def error(x, y_one_hot, W, b, data, num_classes):
    pred_softmax = tf.nn.softmax(tf.matmul(x, W) + b)
    wrong_prediction = tf.not_equal(tf.argmax(pred_softmax, 1), tf.argmax(y_one_hot, 1))
    # Calculate accuracy
    pred_error = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))
    return pred_error.eval({x: data.x, y_one_hot: one_hot(data.y, num_classes)})


def get_cost(cost_name, num_classes, num_sampled, x, y, y_one_hot, W, b, idx, u):
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
        return cost_lt(x, y, W, b, idx, u, num_classes, num_sampled)


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


def cost_lt(x, y, W, b, idx, u, num_classes, num_sampled):
    # Log-tricks

    sgd_weight = float(num_classes) / float(num_sampled)

    def cost_function(labels, logits, num_classes):
        u_idx = tf.transpose(embedding_ops.embedding_lookup(u, idx))
        true_logit = _sum_rows(labels * logits)
        repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
        logit_difference = logits - repeated_true_logit
        u_idx = u_idx.assign(tf.maximum(tf.maximum(logit_difference), u_idx))
        return u_idx + tf.exp(-u_idx) + sgd_weight * _sum_rows((1 - labels) * tf.exp(logit_difference))

        # Multiply the cost by float(num_classes) to account for the uniform sampling
        # return float(num_classes) * _sum_rows((1 - labels) * softplus(logit_difference))

    cost = sampled_cost_wrapper(cost_function, sampler=uniform_candidate_sampler, subtract_log_q=False)

    return cost(weights=tf.transpose(W),
                biases=b,
                labels=y,
                inputs=x,
                num_sampled=num_sampled,
                num_classes=num_classes
                )

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
