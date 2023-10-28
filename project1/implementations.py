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
        w: Optimized weight vector after gradient descent
        loss: The final loss value.
     """
    w = initial_w
    for iter_num in range(max_iters):
        # compute loss and gradient
        gradient_vector = calculate_gradient(y, tx, w)
        # update the weights
        w = w - gamma * gradient_vector
        loss = calculate_mse_loss(y, tx, w)
        #print("Mean Squared Error GD => {0}/{1}: loss={2}".format(iter_num, max_iters - 1, loss))  # for tracking the situation
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    
    Args:
        y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D))
        initial_w: Initial vector of weights for the model. (shape (D, ))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.
    
    Returns:
        w: Optimized weight vector after stochastic gradient descent
        loss: The final loss value.
    """
    w = initial_w

    for n_iter in range(max_iters):
        for yn, txn in batch_iter(y, tx, 1, 1):
            gradient_vector = calculate_gradient(yn, txn, w)
            w = w - gamma * gradient_vector
            loss = calculate_mse_loss(yn, txn, w)
        
        #print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format( bi=n_iter, ti=max_iters - 1, l=loss))  # for tracking the situation
    
    return w, loss


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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic Regression using Gradient Descent.
    
    Args:
        y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D))
        initial_w: Initial vector of weights for the model. (shape (D, ))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.
    
    Returns:
        w: Optimized weight vector.
        loss: The final loss value.
    """
    
    w = initial_w
    for _ in range(max_iters):
        gradient_vector = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        loss = calculate_logistic_loss(y, tx, w)
        
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.
    
    Args:
        y: Vector of labels. (shape (N,))
        tx: Matrix of features/input data. (shape (N,D))
        lambda_: Regularization parameter.
        initial_w: Initial vector of weights for the model. (shape (D, ))
        max_iters: Maximum number of iterations for the optimization.
        gamma: Learning rate.
    
    Returns:
        w: Optimized weight vector.
        loss: The final loss value.
    """
    
    w = initial_w
    for _ in range(max_iters):
        gradient = calculate_logistic_regression_regularized(y, tx, w, lambda_)
        w = w - gamma * gradient
        loss = calculate_logistic_loss(y, tx, w)
        
    return w, loss
#########################################################################################################

def mean_squared_error_gd_weighted(y, tx, initial_w, max_iters, gamma, weights):
    """Linear regression using gradient descent with weighted samples."""
    w = initial_w
    for _ in range(max_iters):
        gradient_vector = calculate_weighted_gradient(y, tx, w, weights)
        w = w - gamma * gradient_vector
        loss = calculate_weighted_mse_loss(y, tx, w, weights)
    return w, loss

def mean_squared_error_sgd_weighted(y, tx, initial_w, max_iters, gamma, weights):
    """Linear regression using stochastic gradient descent with weighted samples."""
    w = initial_w
    for n_iter in range(max_iters):
        for yn, txn, weight_n in zip(y, tx, weights):
            gradient_vector = calculate_weighted_gradient(yn, txn, w, weight_n)
            w = w - gamma * gradient_vector
            loss = calculate_weighted_mse_loss(yn, txn, w, weight_n)
    return w, loss

def least_squares_weighted(y, tx, weights):
    """Linear regression using weighted normal equations."""
    W = np.diag(weights)
    
    # Calculate the terms of the weighted normal equations
    X_transpose_W_X = tx.T.dot(W).dot(tx)
    X_transpose_W_y = tx.T.dot(W).dot(y)

    # Solve for weights w
    w = np.linalg.solve(X_transpose_W_X, X_transpose_W_y)
    loss = calculate_weighted_mse_loss(y, tx, w, weights)
    
    return w, loss

def ridge_regression_weighted(y, tx, lambda_, weights):
    """Ridge regression using weighted normal equations."""
    W = np.diag(weights)
    
    I = np.identity(tx.shape[1])
    X_transpose_W_X = tx.T.dot(W).dot(tx)
    X_transpose_W_y = tx.T.dot(W).dot(y)
    coef_mat = X_transpose_W_X + 2 * len(y) * lambda_ * I
    constant_vector = X_transpose_W_y

    # Solve for weights w
    w = np.linalg.solve(coef_mat, constant_vector)
    loss = calculate_weighted_mse_loss(y, tx, w, weights)
    
    return w, loss


def logistic_regression_weighted(y, tx, initial_w, max_iters, gamma, weights):
    """Logistic Regression using Gradient Descent with weighted samples."""
    w = initial_w
    for _ in range(max_iters):
        gradient_vector = calculate_logistic_gradient_weighted(y, tx, w, weights)
        w = w - gamma * gradient_vector
        loss = calculate_logistic_loss_weighted(y, tx, w, weights)
        
    return w, loss

def reg_logistic_regression_weighted(y, tx, lambda_, initial_w, max_iters, gamma, weights):
    """Regularized logistic regression using gradient descent with weighted samples."""
    w = initial_w
    for _ in range(max_iters):
        gradient = calculate_logistic_gradient_regularized_weighted(y, tx, w, lambda_, weights)
        w = w - gamma * gradient
        loss = calculate_logistic_loss_regularized_weighted(y, tx, w, lambda_, weights)
        
    return w, loss