from helpers import *
import numpy as np


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


def calculate_gradient(y, tx, w):
    """
    This function calculates the gradient at w and return gradient and error values.
    Args:
        tx: A numpy array of shape (N,) containing the observed outputs.
        error:

    Returns:
       The gradient of least squares (shape (D,)) and the error between observed and predicted outputs (shape (N,))
    """
    error = y - tx.dot(w)
    return -tx.T.dot(error) / error.size


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
    return np.exp(x) / (1. + np.exp(x))

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


def calculate_weighted_logistic_loss(y, tx, w, w0, w1):
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
    pred_probs = np.exp(t) / (1 + np.exp(t))
    loss = -np.mean(w1 * y * np.log(pred_probs) + w0 * (1 - y) * np.log(1 - pred_probs))
    return loss

########################## FUNCTIONS #####################################################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
     Linear regression using gradient descent
     Args:
         y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D))
        initial_w: Initial vector of weights for the model. (shape (D, ))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.
     Returns:
        weigth_list[-1]:Optimized weight vector after gradient descent
        loss_values[-1]: The final loss value.
     """
    weigth_list = [initial_w]
    w = initial_w
    loss_values = [calculate_mse_loss(y, tx, w)]
    for iter_num in range(max_iters):
        # compute loss and gradient
        gradient_vector = calculate_gradient(y, tx, w)
        # update the weights
        w = w - gamma * gradient_vector
        loss = calculate_mse_loss(y, tx, w)
        weigth_list.append(w)
        loss_values.append(loss)
        print("Mean Squared Error GD => {0}/{1}: loss={2}".format(iter_num, max_iters - 1,
                                                                  loss))  # for tracking the situation
    return weigth_list[-1], loss_values[-1]

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    """
         Linear regression using stochastic gradient descent
         Args:
             y: Vector of labels. (shape (N,))
            tx: Matrix of features/input data. (shape (N,D))
            initial_w: Initial vector of weights for the model. (shape (D, ))
            max_iters: Maximum number of iterations for the optimization.
            gamma: Learning rate.
         Returns:
            weigth_list[-1]:Optimized weight vector after stochastic gradient descent
            loss_values[-1]: The final loss value.
         """
    w = initial_w
    weigth_list = [initial_w]
    loss_values = [calculate_mse_loss(y, tx, w)]

    for n_iter in range(max_iters):
        for yn, txn in batch_iter(y, tx, 1, 1):
            gradient_vector = calculate_gradient(yn, txn, w)
            w = w - gamma * gradient_vector
            loss = calculate_mse_loss(yn, txn, w)
            weigth_list.append(w)
            loss_values.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))  # for tracking the situation
    return weigth_list[-1], loss_values[-1]

def least_squares(y, tx):
    """
    Linear regression with normal equations
     Args:
         y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D)).
     Returns:
        weigth_list[-1]: Optimized weight vector.
        loss_values[-1]: The final loss value.
    """
    # Calculate the terms of the normal equations
    X_transpose_X = tx.T.dot(tx)
    X_transpose_y = tx.T.dot(y)

    # Solve for weights w
    w = np.linalg.solve(X_transpose_X, X_transpose_y)
    loss = calculate_mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_ ):
    """
    Ridge regression using normal equations
     Args:
         y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D)).
        lambda_: Regularization parameter.
     Returns:
        weigth_list: Optimized weight vector.
        loss_values: The final loss value.
    """

    I = np.identity(tx.shape[1])
    X_transpose_X = tx.T.dot(tx)
    X_transpose_y = tx.T.dot(y)
    coef_mat = X_transpose_X + 2 * len(y) * lambda_ * I #2len(y) = 2N is for deleting 1/2N operation
    constant_vector = X_transpose_y

    # Solve for weights w
    w = np.linalg.solve(coef_mat, constant_vector)
    loss = calculate_mse_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """
    Logistic Regression using Gradient Descent.
     Args:
        y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D))
        initial_w: Initial vector of weights for the model. (shape (D, ))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.

     Returns:
        weigth_list: Optimized weight vector.
        loss_values: The final loss value.
    """

    weigth_list = [initial_w]
    loss_values = []
    w = initial_w
    for _ in range(max_iters):
        gradient_vector = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        loss = calculate_logistic_loss(y, tx, w)

        weigth_list.append(w)
        loss_values.append(loss)
    return weigth_list[-1], loss_values[-1]

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
        Regularized logistic regression using gradient descent
         Args:
            y: Vector of labels. (shape (N,))
            tx: Matrix of features/input data. (shape (N,D))
            initial_w: Initial vector of weights for the model. (shape (D, ))
            max_iters: Maximum number of iterations for the optimization.
            gamma: Learning rate.

         Returns:
            weigth_list[-1]: Optimized weight vector.
            loss_values[-1]: The final loss value.
        """
    weight_list = [initial_w]
    loss_values = []
    w = initial_w

    for _ in range(max_iters):
        predictions = sigmoid(tx.dot(w))
        gradient = calculate_logistic_regression_regularized(y, tx, w, lambda_, predictions)
        w = w - gamma * gradient
        loss = calculate_logistic_loss(y, tx, w)

        weight_list.append(w)
        loss_values.append(loss)

    return weight_list[-1], loss_values[-1]


def calculate_weighted_logistic_gradient_with_regularization(y, tx, w, lambda_, w0, w1):
    """
    Compute the gradient for weighted logistic regression with regularization.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.
        lambda_: Regularization parameter.
        w1: Weight for class 1 (minority class).
        w2: Weight for class 0 (majority class).

    Returns:
        Weighted logistic regression gradient with regularization.
    """
    txw=np.dot(tx, w)
    pred_probs = sigmoid(txw)
    gradient = -np.dot(tx.T,(w1*y*(1-pred_probs))-(w0*(1-y)*pred_probs))/y.shape[0] + lambda_ * w
    #gradient = np.dot(tx.T, (pred_probs - y) * (w1 * y - w2 * (1 - y))) /y.shape[0] + lambda_ * w
    return gradient


import numpy as np

def reg_weighted_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, class_weights=None):
    """
    Regularized weighted logistic regression using gradient descent for binary classification with labels -1 and 1.

    Args:
        y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N, D))
        initial_w: Initial vector of weights for the model. (shape (D,))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.
        class_weights: Class weights for labels -1 and 1 as a tuple (weight_for_minus_1, weight_for_1).

    Returns:
        weight_list[-1]: Optimized weight vector.
        loss_values[-1]: The final loss value.
    """
    weight_list = [initial_w]
    loss_values = []
    w = initial_w

    # Calculate class weights
    if class_weights is None:
        w_0 = 1.0  # Default weight for 0
        w_1 = 1.0  # Default weight for 1
    else:
        w_0, w_1 = class_weights
    
    print(w_0,w_1)
    # Convert labels from -1 and 1 to 0 and 1
    #y = (y + 1) / 2

    for _ in range(max_iters):
        gradient = calculate_weighted_logistic_gradient_with_regularization(y, tx, w, lambda_, w_0, w_1)
        w = w - gamma * gradient
        loss = calculate_weighted_logistic_loss(y, tx, w, w_0, w_1)
        #print(loss)
        weight_list.append(w)
        loss_values.append(loss)
    print(loss_values[-5:])
    return weight_list[-1], loss_values[-1]
