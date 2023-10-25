
import sys
sys.path.append('/content/ML_course/projects/project1/')
from helpers import *



print(x_train[:5])

import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

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
    train_features, _ = standardize(train_features)
    test_features, _ = standardize(test_features)

    # Apply polynomial features if degree > 1
    if polynomial_degree > 1:
        train_features = build_polynomial_features(train_features, polynomial_degree)
        test_features = build_polynomial_features(test_features, polynomial_degree)

    # Add a column of ones for bias/intercept
    train_features = np.c_[np.ones(train_features.shape[0]), train_features]
    test_features = np.c_[np.ones(test_features.shape[0]), test_features]

    return train_features, test_features