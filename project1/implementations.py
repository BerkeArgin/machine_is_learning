from helpers import *
import numpy as np

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
    loss_values = []
    w = initial_w
    for iter_num in range(max_iters):
        loss = calculate_mse_loss(y, tx, w)
        gradient_vector = calculate_gradient(tx, loss)
        w = w - gamma * gradient_vector
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
    weigth_list = [initial_w]
    loss_values = []
    w = initial_w
    for n_iter in range(max_iters):
        for yn, txn in batch_iter(y, tx, 1, 1):
            loss = calculate_mse_loss(yn, txn, w)
            gradient_vector = calculate_gradient(txn, loss)
            w = w - gamma * gradient_vector
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
        loss = calculate_logistic_loss(y, tx, w)
        w = w - gamma * gradient_vector

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