import csv
import numpy as np
import os


def calculate_mse_loss(y, tx, w):
    """
    Calculate Mean Absolute Error Loss
    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
       The Mean Squared Error Loss between observed and predicted outputs

    """
    y_pred = tx.dot(w)
    return np.mean((y - y_pred) ** 2) / 2


def calculate_gradient(tx, error):
    """
    This function calculates the gradient at w and return gradient and error values.
    Args:
        tx: A numpy array of shape (N,) containing the observed outputs.
        error:

    Returns:
       The gradient of least squares (shape (D,)) and the error between observed and predicted outputs (shape (N,))
    """
    gradient = -np.dot(tx.T, error) / error.size
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Args:
        y:  The output desired values.
        tx: The input data.
        batch_size: Size of each mini-batch.
        num_batches: The number of mini-batches to produce.
        shuffle: If True, the data is shuffled; otherwise, data is taken sequentially.

    Returns:
       Containing the output labels and input data for a mini-batch.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
            data_size - max_batches * batch_size
    )
    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            # Without an offset, batches might start at indices: [0, 32, 64, 96] (assume remainder=4)
            # With a random offset, batches would start at: [4, 36, 68, 100]
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size
    for start in idxs:
        start_index = start
        end_index = (
                start_index + batch_size
        )
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(x):
    """
    Applies the sigmoid function to a given input.
    Args:
        x: Given input.
    Returns:
        The value of sigmoid function.
    """
    return 1. / (1. + np.exp(-x))

def calculate_logistic_loss(y, tx, w):
    """
    Compute the negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
        Negative log likelihood loss.
    """
    t = np.dot(tx, w)
    loss = np.sum(np.log(1 + np.exp(t)) - y * t) / y.shape[0]
    return loss


def calculate_weighted_logistic_loss(y, tx, w, w1, w2):
    """
    Compute the weighted negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.
        w1: Weight for class 1 (minority class).
        w2: Weight for class 0 (majority class).

    Returns:
        Weighted negative log likelihood loss.
    """
    t = np.dot(tx, w)
    pred_probs = 1 / (1 + np.exp(-t))
    loss = -np.mean(w1 * y * np.log(pred_probs + 1e-15) + w2 * (1 - y) * np.log(1 - pred_probs + 1e-15))
    return loss

def calculate_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
        gradient_vector: Gradient vector which has shape (D,)
    """
    predicted_probs = sigmoid(tx.dot(w))
    gradient_vector = tx.T.dot(predicted_probs - y)
    return gradient_vector

def calculate_logistic_regression_regularized(y, tx, w, lambda_, predictions):
    """
        Compute the gradient of the negative log likelihood for logistic regression with regularization term for l2".

        Args:
            y: A numpy array of shape (N,) containing the observed outputs.
            tx: A numpy array of shape (N, D) containing the feature matrix of the data.
            w: A numpy array of shape (D,) which is the weight vector.
            lambda_: Regularization parameter.
            predictions: The result of sigmoid function.

        Returns:
            gradient_vector_regularized: Gradient vector which has shape (D,)
        """
    gradient_vector_regularized = tx.T.dot(predictions - y) / y.shape[0] + 2 * lambda_ * w
    return gradient_vector_regularized