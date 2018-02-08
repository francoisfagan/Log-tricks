"""Multinomial Regression Loss variables and cost functions are defined here.

"""
import numpy as np
from scipy.special import lambertw
# from implicit_cy import f1
from scipy.optimize import brentq


def error_log_loss(logits, y):
    pred = np.argmax(logits, axis=1)  # Dimensions [data_size]

    max_logit = logits[list(range(len(pred))), pred]  # Dimensions [data_size]
    logits = logits - max_logit[:, None]  # Log-max trick, Dimensions [data_size] x [num_classes]
    log_loss = np.mean(- logits[list(range(len(pred))), y]
                       + np.log(np.sum(np.exp(logits), axis=1)))  # Dimensions [1]

    mean_error = np.mean(y != pred)
    return [mean_error, log_loss]


class SGD:
    def __init__(self, dim, num_classes, num_train_points):
        self.num_classes = num_classes
        self.W = np.zeros((dim, num_classes))  # Dimension: [num_classes] x [dim]
        self.u = np.ones(num_train_points) * 0.0  # np.log(num_classes)  # Dimension: [num_train_points]

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
        logits = data.x_sparse.dot(self.W)  # Dimensions [data_size] x [num_classes]
        return error_log_loss(logits, data.y)


class Softmax(SGD):
    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """
        x_row, x_row_idx, x_col_idx, x_val = x

        # Calculate logits
        # Since x is sparse, this computation is as fast as it can be
        logits = x_row * self.W  # Dimensions [batch_size] x [num_classes]

        # Log-max trick to make numerically stable
        logits = logits - np.max(logits, axis=1)[:, None]  # Dimensions [batch_size] x [num_classes]

        # Take exponents and normalize
        coef = np.exp(logits)  # Dimensions [batch_size] x [num_classes]
        coef = coef / np.sum(coef, axis=1)[:, None]  # Dimensions [batch_size] x [num_classes]

        # Minus 1 for true classes
        coef[[list(range(len(y))), y]] -= 1.0  # Dimensions [batch_size] x [num_classes]

        # Take SGD step
        grad = coef[x_row_idx, :] * x_val[:, None]  # Dimensions [non-zero entries in x] x [num_classes]
        self.W[x_col_idx, :] -= learning_rate * grad


class VanillaSGD(SGD):
    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """

        x_row, x_row_idx, x_col_idx, x_val = x

        # Find batch_size and num_sampled
        sampled_classes = np.setdiff1d(sampled_classes, y)
        num_sampled = len(sampled_classes)

        # Calculate logit_difference = x_i^\top(w_k-w_{y_i}) for all i in idx and k in sampled_classes
        logit_sampled = x_row.dot(self.W[:, sampled_classes])  # Dimensions [num_sampled]
        logit_true = x_row.dot(self.W[:, y])[0, 0]  # Dimensions [1]
        logit_diff = logit_sampled - logit_true  # Dimensions [num_sampled]

        # Gradient coefficients
        scaling = (self.num_classes - 1) / num_sampled
        coef = scaling * np.exp(logit_diff - self.u[idx])
        # Dimensions [num_sampled]

        # SGD gradients
        u_grad = 1 - np.exp(-self.u[idx]) - np.sum(coef, axis=1)  # Dimensions [1]
        w_sample_grad = coef[x_row_idx, :] * x_val[:, None]  # Dimensions [non-zero entries in x] x [num_samples]
        w_true_grad = - np.sum(w_sample_grad, axis=1)  # Dimensions [non-zero entries in x]
        # https://stackoverflow.com/questions/5795700/multiply-numpy-array-of-scalars-by-array-of-vectors

        # Update variables
        self.u[idx] -= learning_rate * u_grad
        self.u[idx] = max(0.0, self.u[idx])

        # Update sampled W
        W_sampled_indices = np.vstack((np.tile(x_col_idx, num_sampled),
                                       np.repeat(sampled_classes, len(x_col_idx)))).tolist()
        # Dimensions [non-zero entries in x * num_sampled]
        grad_sampled_indices = np.vstack((np.tile(np.arange(len(x_col_idx)), num_sampled),
                                          np.repeat(np.arange(num_sampled), len(x_col_idx)))).tolist()
        # Dimensions [non-zero entries in x * num_sampled]
        self.W[W_sampled_indices] -= learning_rate * w_sample_grad[grad_sampled_indices]

        # Update true W
        self.W[x_col_idx, y] -= learning_rate * w_true_grad


class Umax(SGD):
    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """
        x_row, x_row_idx, x_col_idx, x_val = x

        # Find batch_size and num_sampled
        sampled_classes = np.setdiff1d(sampled_classes, y)
        num_sampled = len(sampled_classes)

        # Calculate logit_difference = x_i^\top(w_k-w_{y_i}) for all i in idx and k in sampled_classes
        logit_sampled = x_row.dot(self.W[:, sampled_classes])  # Dimensions [num_sampled]
        logit_true = x_row.dot(self.W[:, y])[0, 0]  # Dimensions [1]
        logit_diff = logit_sampled - logit_true  # Dimensions [num_sampled]

        # Update u
        logit_diff_max = np.max(logit_diff, axis=1)  # Dimensions [1]
        if logit_diff_max < 0:
            u_bound = np.log(1.0 + np.sum(np.exp(logit_diff), axis=1))
        else:
            u_bound = logit_diff_max + np.log(np.exp(-logit_diff_max) +
                                              np.sum(np.exp((logit_diff - logit_diff_max)), axis=1))  # Dimensions [1]
        if self.u[idx] < (u_bound - 1):
            self.u[idx] = u_bound  # Dimensions [1]

        # Gradient coefficients
        scaling = (self.num_classes - 1) / num_sampled
        coef = scaling * np.exp(logit_diff - self.u[idx])
        # Dimensions [num_sampled]

        # SGD gradients
        u_grad = 1 - np.exp(-self.u[idx]) - np.sum(coef, axis=1)  # Dimensions [1]
        w_sample_grad = coef[x_row_idx, :] * x_val[:, None]  # Dimensions [non-zero entries in x] x [num_samples]
        w_true_grad = - np.sum(w_sample_grad, axis=1)  # Dimensions [non-zero entries in x]
        # https://stackoverflow.com/questions/5795700/multiply-numpy-array-of-scalars-by-array-of-vectors

        # Update variables
        self.u[idx] -= learning_rate * u_grad
        self.u[idx] = max(0.0, self.u[idx])

        # Update sampled W
        W_sampled_indices = np.vstack((np.tile(x_col_idx, num_sampled),
                                       np.repeat(sampled_classes, len(x_col_idx)))).tolist()
        # Dimensions [non-zero entries in x * num_sampled]
        grad_sampled_indices = np.vstack((np.tile(np.arange(len(x_col_idx)), num_sampled),
                                          np.repeat(np.arange(num_sampled), len(x_col_idx)))).tolist()
        # Dimensions [non-zero entries in x * num_sampled]
        self.W[W_sampled_indices] -= learning_rate * w_sample_grad[grad_sampled_indices]

        # Update true W
        self.W[x_col_idx, y] -= learning_rate * w_true_grad


class tilde_Umax(SGD):
    """
    Alternative double sum formulation
    """

    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """
        x_row, x_row_idx, x_col_idx, x_val = x

        # Find batch_size and num_sampled
        sampled_classes = np.setdiff1d(sampled_classes, y)
        num_sampled = len(sampled_classes)

        # Calculate logit_difference = x_i^\top(w_k-w_{y_i}) for all i in idx and k in sampled_classes
        logit_sampled = x_row.dot(self.W[:, sampled_classes])[0]  # Dimensions [num_sampled]
        logit_true = x_row.dot(self.W[:, y])[0, 0]  # Dimensions [1]
        logit_true_sampled = np.concatenate((np.array([logit_true]), logit_sampled))

        # Update u
        logit_max = np.max(logit_true_sampled)  # Dimensions [1]
        u_bound = logit_max + np.log(np.sum(np.exp((logit_true_sampled - logit_max))))  # Dimensions [1]
        if self.u[idx] < (u_bound - 1):
            self.u[idx] = u_bound  # Dimensions [1]

        # Gradient coefficients
        scaling = (self.num_classes - 1) / num_sampled
        coef_sampled = scaling * np.exp(logit_sampled - self.u[idx])[None, :]
        coef_true = np.exp(logit_true - self.u[idx])

        # SGD gradients
        u_grad = 1 - np.exp(coef_true) - np.sum(coef_sampled)  # Dimensions [1]
        w_sample_grad = coef_sampled[x_row_idx, :] * x_val[:,
                                                     None]  # Dimensions [non-zero entries in x] x [num_samples]
        w_true_grad = coef_true - 1  # Dimensions [non-zero entries in x]
        # https://stackoverflow.com/questions/5795700/multiply-numpy-array-of-scalars-by-array-of-vectors

        # Update variables
        self.u[idx] -= learning_rate * u_grad

        # Update sampled W
        W_sampled_indices = np.vstack((np.tile(x_col_idx, num_sampled),
                                       np.repeat(sampled_classes, len(x_col_idx)))).tolist()
        # Dimensions [non-zero entries in x * num_sampled]
        grad_sampled_indices = np.vstack((np.tile(np.arange(len(x_col_idx)), num_sampled),
                                          np.repeat(np.arange(num_sampled), len(x_col_idx)))).tolist()

        # if w_sample_grad.shape == (1, 1):
        #     print(w_sample_grad)
        #     print(grad_sampled_indices)
        #     print(w_sample_grad.shape)
        #     print(w_sample_grad[grad_sampled_indices].shape)
        #     print(self.W[W_sampled_indices].shape)
        # Dimensions [non-zero entries in x * num_sampled]
        self.W[W_sampled_indices] -= learning_rate * w_sample_grad[grad_sampled_indices]

        # Update true W
        self.W[x_col_idx, y] -= learning_rate * w_true_grad


class Implicit(SGD):

    def a(self, u_temp, multiplier):
        """Returns solution to a as defined in the paper"""
        diff = multiplier - u_temp
        return robust_lambert_w_exp(diff)

    def f1(self, u_temp, multiplier, x_norm, learning_rate, u_old):
        """Zeroth derivative of u equation"""
        diff = multiplier - u_temp
        if diff > 15:
            # Asymptotic expansion from equations (15-18) of
            # http://mathworld.wolfram.com/LambertW-Function.html
            L1 = diff
            L2 = np.log(diff)
            a_temp = (L1
                      - L2
                      + L2 / L1
                      + L2 * (-2 + L2) / (2 * L1 ** 2)
                      + L2 * (6 - 9 * L2 + 2 * L2 ** 2) / (6 * L1 ** 3)
                      + L2 * (-12 + 36 * L2 - 22 * L2 ** 2 + 3 * L2 ** 3) / (12 * L1 ** 4)
                      )
        else:
            a_temp = np.real(lambertw(np.exp(diff))) #, tol=1e-5
        return (2 * learning_rate
                - 2 * learning_rate * np.exp(-u_temp)
                - a_temp / (1 + a_temp) / x_norm
                + 2 * (u_temp - u_old)
                - a_temp ** 2 / (1 + a_temp) / x_norm
                )

    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """

        # Unpack the data
        x_row, x_row_idx, x_col_idx, x_val = x

        # Calculate values to feed into the f1 function
        x_norm = x_val.dot(x_val)
        x_dot_W = x_row.dot(self.W[:, sampled_classes] - self.W[:, y])[0, 0]
        multiplier = (x_dot_W + np.log(2 * learning_rate * (self.num_classes - 1) * x_norm))

        # Default values for
        u_optimal = 1.0
        a_optimal = 1.0

        # Calculate upper and lower bounds for Brent's method
        small_argument = (x_dot_W - self.u[idx][0]) < 15
        if self.f1(self.u[idx], multiplier, x_norm, learning_rate, self.u[idx]) < 0:
            bounds = (self.u[idx][0],
                      self.u[idx][0] - learning_rate
                      + (np.real(lambertw(learning_rate
                                          * np.exp(learning_rate - self.u[idx][0])
                                          * (1 + (self.num_classes - 1)
                                             * np.exp(x_dot_W))))
                         if small_argument else
                         (np.log(learning_rate)
                          + learning_rate - self.u[idx][0]
                          + np.log(self.num_classes - 1)
                          + x_dot_W)
                         )
                      )
        else:
            bounds = (max(0.0,
                          self.u[idx][0] - learning_rate
                          + np.real(lambertw(learning_rate
                                             * np.exp(learning_rate - self.u[idx][0])
                                             * (1 + (self.num_classes - 1)
                                                * np.exp(x_dot_W - 2 * learning_rate * x_norm))))
                          ),
                      self.u[idx][0]
                      )

        # Avoid rounding errors
        round_error = 1e-5
        bounds = (bounds[0] - round_error, bounds[1] + round_error)

        # Calculate optimal u and a values
        u_optimal = brentq(self.f1,
                           bounds[0], bounds[1],
                           args=(multiplier, x_norm, learning_rate, self.u[idx]),
                           )
        a_optimal = self.a(u_optimal, multiplier)

        # Update variables
        self.u[idx] = max(0.0, u_optimal)
        self.W[x_col_idx, sampled_classes] -= a_optimal * x_val / (2 * x_norm)
        self.W[x_col_idx, y] += a_optimal * x_val / (2 * x_norm)


def robust_lambert_w_exp(x):
    # Returns lambert_w(exp(x))
    if x > 15:
        # Asymptotic expansion from equations (15-18) of
        # http://mathworld.wolfram.com/LambertW-Function.html
        L1 = x
        L2 = np.log(x)
        return (L1
                - L2
                + L2 / L1
                + L2 * (-2 + L2) / (2 * L1 ** 2)
                + L2 * (6 - 9 * L2 + 2 * L2 ** 2) / (6 * L1 ** 3)
                + L2 * (-12 + 36 * L2 - 22 * L2 ** 2 + 3 * L2 ** 3) / (12 * L1 ** 4)
                )
    else:
        return np.real(lambertw(np.exp(x)))


class Implicit_simple(SGD):

    def update(self, x, y, idx, sampled_classes, learning_rate):
        """
        The dimensions of the matrices below is [batch_size] x [num_classes] unless otherwise stated
        """
        x_row, x_row_idx, x_col_idx, x_val = x

        # Update u
        t1 = learning_rate * np.exp(-self.u[idx][0] + learning_rate)
        t2 = (x_row.dot(self.W[:, sampled_classes] - self.W[:, y]) - self.u[idx]
              + learning_rate
              + np.log(learning_rate * (self.num_classes - 1)))[0, 0]
        log_b = np.log(t1 + np.exp(t2)) if t2 < 15 else t2
        self.u[idx] = self.u[idx] - learning_rate + robust_lambert_w_exp(log_b)
        self.u[idx] = max(0.0, self.u[idx])

        # Update w
        x_norm = x_val.dot(x_val)
        t = (x_row.dot(self.W[:, sampled_classes] - self.W[:, y])
             + np.log(2 * learning_rate * (self.num_classes - 1) * x_norm))[0, 0]
        a = robust_lambert_w_exp(t)
        self.W[x_col_idx, sampled_classes] -= a * x_val / (2 * x_norm)
        self.W[x_col_idx, y] += a * x_val / (2 * x_norm)
