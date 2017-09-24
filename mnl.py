"""Multinomial Regression Loss variables and cost functions are defined here.

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn_impl import _compute_sampled_logits, _sum_rows, sigmoid_cross_entropy_with_logits
from tensorflow.python.ops import nn_ops, embedding_ops


def one_hot(y, num_classes):
    return np.eye(num_classes)[y[:, 0]]


def graph(dim, num_classes, num_train_points):
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, dim])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.int64, [None, 1])  # 0-9 digits recognition => 10 classes
    y_one_hot = tf.placeholder(tf.float32, [None, num_classes])  # 0-9 digits recognition => 10 classes
    idx = tf.placeholder(tf.int64, [None, 1])  # data point indices

    # Set model weights
    W = tf.Variable(tf.zeros([dim, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    u = tf.Variable(tf.ones([num_train_points]) * tf.log(float(num_classes)))  # Initialize u_i = log(K)

    variables = [x, y, y_one_hot, W, b, idx, u]

    return variables


def get_cost(cost_name, x, y, y_one_hot, W, b, idx, u):
    if cost_name == 'softmax': return cost_softmax(x, y_one_hot, W, b)


def cost_softmax(x, y_one_hot, W, b):
    # Softmax without sampling
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y_one_hot * tf.log(pred), reduction_indices=1))
    return cost


def error(x, y_one_hot, W, b, data, num_classes):
    pred_softmax = tf.nn.softmax(tf.matmul(x, W) + b)
    wrong_prediction = tf.not_equal(tf.argmax(pred_softmax, 1), tf.argmax(y_one_hot, 1))
    # Calculate accuracy
    error = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))
    return error.eval({x: data.x, y_one_hot: one_hot(data.y, num_classes)})




    # def stable_logistic(x):
    #     """Calculates log(1+exp(x)) in a stable way.
    #     https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    #     """
    #     return tf.maximum(x, 0.0) + tf.log(1.0 + tf.exp(-tf.abs(x)))
    #
    # def OVE(labels, logits):
    #     true_logit = _sum_rows(labels * logits)
    #     repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
    #     logit_difference = logits - repeated_true_logit
    #     return _sum_rows((1 - labels) * stable_logistic(logit_difference))
    #
    # def my_sigmoid_cross_entropy_with_logits(labels, logits):
    #     """My implementation of nn_ops.softmax_cross_entropy_with_logits
    #     Used to make sure I can do this right"""
    #     return _sum_rows(tf.maximum(logits, 0.0) - logits * labels + tf.log(1.0 + tf.exp(-abs(logits))))
    #
    # def custom_sampled_loss(custom_loss_function):
    #     def loss(weights,
    #              biases,
    #              labels,
    #              inputs,
    #              num_sampled,
    #              num_classes,
    #              num_true=1,
    #              sampled_values=None,
    #              remove_accidental_hits=True,
    #              partition_strategy="mod",
    #              name="ove_loss"):
    #         logits, labels = _compute_sampled_logits(
    #             weights=weights,
    #             biases=biases,
    #             labels=labels,
    #             inputs=inputs,
    #             num_sampled=num_sampled,
    #             num_classes=num_classes,
    #             num_true=num_true,
    #             sampled_values=sampled_values,
    #             subtract_log_q=False,
    #             remove_accidental_hits=remove_accidental_hits,
    #             partition_strategy=partition_strategy,
    #             name=name)
    #         return custom_loss_function(labels=labels, logits=logits)
    #
    #     return loss
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
    # # Sampled softmax = Importance sampling
    # cost_sampled_softmax = tf.nn.sampled_softmax_loss(weights=tf.transpose(W),
    #                                                   biases=b,
    #                                                   inputs=x,
    #                                                   labels=y_int,
    #                                                   num_sampled=5,
    #                                                   num_classes=10)
    #
    # # Noise Contrastive Estimation
    # cost_nce = tf.nn.nce_loss(weights=tf.transpose(W),
    #                           biases=b,
    #                           inputs=x,
    #                           labels=y_int,
    #                           num_sampled=5,
    #                           num_classes=10)
    #
    # # One vs Each
    # cost_ove = custom_sampled_loss(OVE)(weights=tf.transpose(W),
    #                                     biases=b,
    #                                     inputs=x,
    #                                     labels=y_int,
    #                                     num_sampled=5,
    #                                     num_classes=10)
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
    #
    # def ld_loss(weights,
    #             biases,
    #             datapoint_weights,
    #             labels,
    #             inputs,
    #             idx,
    #             num_sampled,
    #             num_classes,
    #             num_true=1,
    #             sampled_values=None,
    #             remove_accidental_hits=True,
    #             partition_strategy="mod",
    #             name="ld_loss"):
    #     logits, labels = _compute_sampled_logits(
    #         weights=weights,
    #         biases=biases,
    #         labels=labels,
    #         inputs=inputs,
    #         num_sampled=num_sampled,
    #         num_classes=num_classes,
    #         num_true=num_true,
    #         sampled_values=sampled_values,
    #         subtract_log_q=False,
    #         remove_accidental_hits=remove_accidental_hits,
    #         partition_strategy=partition_strategy)
    #
    #     sampled_dp_weight = tf.transpose(embedding_ops.embedding_lookup(
    #         datapoint_weights, idx, partition_strategy=partition_strategy))
    #
    #     true_logit = _sum_rows(labels * logits)
    #     repeated_true_logit = tf.tile(tf.reshape(true_logit, [-1, 1]), [1, tf.shape(logits)[1]])
    #     logit_difference = logits - repeated_true_logit
    #
    #     # - sampled_dp_weight + tf.exp(sampled_dp_weight) *
    #     #
    #     # (1.0 + _sum_rows((1 - labels) * stable_logistic(logit_difference)))
    #     return sampled_dp_weight + tf.exp(-sampled_dp_weight) * _sum_rows(
    #         (1 - labels) * stable_logistic(logit_difference))
    #
    # # Learned Denominator
    # cost_ld = ld_loss(weights=tf.transpose(W),
    #                   biases=b,
    #                   datapoint_weights=u,
    #                   inputs=x,
    #                   idx=idx,
    #                   labels=y_int,
    #                   num_sampled=5,
    #                   num_classes=10)
    #
    # def cost():
    #
    # def accuracy():
