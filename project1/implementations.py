from helpers import *
import numpy as np

########################## PREPROCESSING #####################################################

def standardize(x,mean=None,std=None):
    """Standardize the original data set."""
    if mean==None:
        mean = np.mean(x)
        x = x - mean
        std= np.std(x)
        x = x / std
    else:
        x=x-mean
        x=x/std
    return x, mean, std

def missing_values_outliers(x, coefficient):
    """ Handles missing values by replacing them with the mean of their corresponding column
     and addresses potential outliers by adjusting them into a specified range """
    cp_x = x.copy()
    cp_x[cp_x == -999] = np.nan
    mean = np.nanmean(cp_x, axis=0)
    col_len = cp_x.shape[1]
    for i in range(col_len):
        cp_x[np.isnan(cp_x[:, i]), i] = mean[i]

    if coefficient != -1:
        std_x = np.nanstd(cp_x, axis=0)
        lower_bound = mean - coefficient * std_x
        upper_bound = mean + coefficient * std_x

        cp_x = np.clip(cp_x, lower_bound, upper_bound)
    return cp_x

def remove_highly_correlated_features(X_train, X_test, threshold=0.95):
    """
    Remove one of the columns that has a correlation higher than the specified threshold.
        X_train: Features of the training data (N1, D).
        X_test: Features of the testing data (N2, D).
        threshold: Correlation threshold for feature removal (default is 0.95).
    Returns:
        reduced_X_train: The resulting training data features with correlated features removed.
        reduced_X_test: The resulting testing data features with correlated features removed.
    """
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    high_correlation_indices = np.where(np.triu(np.abs(corr_matrix) > threshold, k=1)) #triu is used since correlations are symmetric
    columns_to_remove = set()

    # Identify which columns to remove
    for i, j in zip(*high_correlation_indices):
        col_i_corr_sum = np.sum(np.abs(corr_matrix[i]))
        col_j_corr_sum = np.sum(np.abs(corr_matrix[j]))
        # Remove the column with the lower total correlation
        if col_i_corr_sum > col_j_corr_sum:
            columns_to_remove.add(j)
        else:
            columns_to_remove.add(i)

    reduced_X_train = np.delete(X_train, list(columns_to_remove), axis=1)
    reduced_X_test = np.delete(X_test, list(columns_to_remove), axis=1) if X_test is not None else None

    return reduced_X_train, reduced_X_test

def add_log_transform(x, log_columns=[]):
    """
    Add the natural logarithm (log) value to specified columns in the input array.
    Args:
        x: numpy array of shape (N, D), where N is the number of samples.
        log_columns: Optional list of column indices to apply the log transformation.

    Returns:
        transformed_x: numpy array of shape (N, D), where the log transformation has been applied to specified columns.
    """
    transformed_x = x.copy()
    #If columns are not specifies apply to all columns
    if not log_columns:
        log_columns = np.arange(x.shape[1])

    for column_index in log_columns:
        valid_data_mask = transformed_x[:, column_index] != -999
        valid_data = transformed_x[valid_data_mask, column_index].copy()
        valid_data[valid_data <= 0] = 1
        transformed_x[valid_data_mask, column_index] = np.log(valid_data)
    #If a value is not valid, it is set to 1 to avoid undefined logarithm operations
    return transformed_x

def build_polynomial_features(input_data, degree, selected_columns=None):
    """
    Build a feature matrix with polynomial basis functions for the input data.
    Args:
        input_data: A numpy array of shape (N, D), where N is the number of samples.
        degree: An integer specifying the maximum degree of polynomial features.
        selected_columns: An optional list of column indices to which the polynomial features will be applied.
                          If None, all columns are considered.
    Returns:
        polynomial_features: A numpy array of shape (N, D_new), where D_new is the total number of polynomial features.
    """
    num_samples, num_columns = input_data.shape

    if selected_columns is None:
        selected_columns = np.arange(num_columns)

    polynomial_features = []

    for col_index in selected_columns:
        column_data = input_data[:, col_index]
        column_features = []

        for d in range(1, degree + 1):
            column_features.append(np.power(column_data, d))

        polynomial_features.append(np.column_stack(column_features))

    polynomial_features = np.column_stack(polynomial_features)

    return polynomial_features

def apply_preprocessing(train_features, test_features, correlation_tolerance=0.01, outlier_coefficient=2.0, polynomial_degree=1, log_transform_columns=[]):
    """
    Apply preprocessing functions to input data.
    """
    # Identify unnecessary columns with only one unique value
    unnecessary_columns = [i for i in range(train_features.shape[1]) if len(np.unique(train_features[:, i])) == 1]

    # Apply log transformation to specified columns
    if log_transform_columns:
        train_features = add_log_transform(train_features, log_transform_columns)
        test_features = add_log_transform(test_features, log_transform_columns)

    # Remove unnecessary columns
    train_features = np.delete(train_features, unnecessary_columns, axis=1)
    test_features = np.delete(test_features, unnecessary_columns, axis=1)

    # Handle missing and outlier values
    train_features = missing_values_outliers(train_features, outlier_coefficient)
    test_features = missing_values_outliers(test_features, outlier_coefficient)

    # Standardize columns
    train_features, _, _ = standardize(train_features)
    test_features, _, _ = standardize(test_features)

    # Apply polynomial features if degree > 1
    if polynomial_degree > 1:
        train_features = build_polynomial_features(train_features, polynomial_degree)
        test_features = build_polynomial_features(test_features, polynomial_degree)

    # Add a column of ones for bias/intercept
    train_features = np.c_[np.ones(train_features.shape[0]), train_features]
    test_features = np.c_[np.ones(test_features.shape[0]), test_features]

    return train_features, test_features

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
    pred_probs = np.exp(txw) / (1 + np.exp(txw))
    gradient = -np.dot(tx.T,(w1*y*(1-pred_probs))-(w0*(1-y)*pred_probs))/y.shape[0] +lambda_ * w
    #gradient = np.dot(tx.T, (pred_probs - y) * (w1 * y - w2 * (1 - y))) /y.shape[0] + lambda_ * w
    return gradient


import numpy as np

def reg_weighted_logistic_regression_balanced(y, tx, lambda_, initial_w, max_iters, gamma, class_weights=None):
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
