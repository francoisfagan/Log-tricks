"""Multinomial Regression Loss variables and cost functions are defined here.

"""
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, find


class SGD:
    def __init__(self, dim, num_classes, num_train_points):
        self.W = np.zeros((dim, num_classes))  # Dimension: [num_classes] x [dim]
        # self.W = csc_matrix((dim, num_classes))  # Dimension: [num_classes] x [dim]
        self.u = np.zeros(num_train_points)  # Dimension: [num_train_points]
        self.num_classes = num_classes

    def update(self, x, y, idx, sampled_classes, learning_rate):
        """ Performs sgd update of variables

        :param x: np.array of dimensions [batch_size] x [dim]
        :param y: np.array of dimensions [batch_size]
        :param idx: np.array of dimensions [batch_size]
        :param sampled_classes: np.array of dimensions [num_sampled]
        :param learning_rate: scalar
        :return:
        """

    def error(self, data):
        pred = np.argmax(data.x.dot(self.W), axis=1)
        mean_error = np.mean(data.y != pred)
        return mean_error


class Softmax(SGD):
    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """

        logits = x * self.W
        #logits = logits.toarray()
        # print(type(logits))
        # print('x.shape', x.shape)
        # print('self.W.shape', self.W.shape)
        # print('x', x)
        # print('W', self.W)
        # print('logits.shape', logits.shape)
        # print('printing logits: ', logits)
        # logits = logits.toarray()
        # print(type(logits))
        # print(logits.shape)

        # Log-max trick to make numerically stable
        logits = logits - np.max(logits, axis=1)[:, None]

        # Take exponents and normalize
        exp_logits = np.exp(logits)
        exp_logits = exp_logits / np.sum(exp_logits, axis=1)[:, None]

        # Minus 1 for true classes
        exp_logits[[list(range(len(y))), y]] -= 1.0

        # Take SGD step
        # grad = exp_logits.T.dot(xx)
        # grad = x.T.dot(exp_logits)
        # self.W = self.W - learning_rate * grad

        x_row_idx, x_col_idx, x_val = find(x)
        grad = (exp_logits[x_row_idx, :] * x_val[:, None])
        self.W[x_col_idx, :] = self.W[x_col_idx, :] - learning_rate * grad


class LogTricks(SGD):
    def update(self, x, y, idx, sampled_classes, learning_rate):
        # Find batch_size and num_sampled
        batch_size = len(idx)
        num_sampled = len(sampled_classes)

        # Calculate logit_difference = x_i^\top(w_k-w_{y_i}) for all i in idx and k in sampled_classes
        logits_sampled = x.dot(self.W[:, sampled_classes])  # Dimensions [batch_size] x [num_sampled]
        logits_true = np.array([np.dot(x[i, :], self.W[:, y[i]])
                                for i in range(batch_size)])  # Dimensions [batch_size]
        logit_true_matrix = np.tile(logits_true[:, None], (1, num_sampled))  # Dimensions [batch_size] x [num_sampled]
        logit_diff = logits_sampled - logit_true_matrix  # Dimensions [batch_size] x [num_sampled]

        # Calculate whether the sampled labels coincide with the true labels
        # labels[i,j] = I(y[i] == s_c[j])
        # Dimensions [batch_size] x [num_sampled]
        labels = (np.tile(sampled_classes, (batch_size, 1)) == np.tile(y, (1, num_sampled))).astype('float')
        # Remove sample from count if it equals the true label
        num_non_true_sampled = np.sum(1 - labels, axis=1)  # Dimensions [batch_size]

        # Update u
        logit_diff_max = np.max(logit_diff, axis=1)  # Dimensions [batch_size]
        logit_diff_max_matrix = np.tile(logit_diff_max[:, None],
                                        (1, num_sampled))  # Dimensions [batch_size] x [num_sampled]
        u_bound = logit_diff_max + np.log(np.exp(-logit_diff_max) +
                                          np.sum((1 - labels) * np.exp((logit_diff - logit_diff_max_matrix)),
                                                 axis=1))  # Dimensions [batch_size]
        self.u[idx] = np.maximum(self.u[idx], u_bound)  # Dimensions [batch_size]

        # SGD step
        scaling_factor = float(self.num_classes) / num_non_true_sampled  # Dimensions [batch_size]
        scaling_factor_matrix = np.tile(scaling_factor[:, None],
                                        (1, num_sampled))  # Dimensions [batch_size] x [num_sampled]
        u_idx_matrix = np.tile(self.u[idx][:, None], (1, num_sampled))  # Dimensions [batch_size] x [num_sampled]
        factor = scaling_factor_matrix * (1 - labels) * np.exp(
            logit_diff - u_idx_matrix)  # Dimensions [batch_size] x [num_sampled]
        u_grad = 1 - np.exp(-self.u[idx]) - np.sum(factor, axis=1)  # Dimensions [batch_size]
        w_sample_grad = np.dot(factor.T, x)  # Dimensions [num_sampled] x [dim]
        w_true_grad = -x * np.sum(factor, axis=1)[:, None]  # Dimensions [batch_size] x [dim]
        # https://stackoverflow.com/questions/5795700/multiply-numpy-array-of-scalars-by-array-of-vectors

        # Update variables
        self.u[idx] -= learning_rate * u_grad
        self.W[:, sampled_classes] -= learning_rate * w_sample_grad.T
        self.W[:, y] -= learning_rate * w_true_grad.T
