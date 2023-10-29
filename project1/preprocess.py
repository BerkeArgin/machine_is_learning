from helpers import *
import numpy as np


def standardize(x,mean=None,std=None):
    """Standardize the original data set."""
    if mean is None or std is None:
        mean = np.mean(x,axis=0)
        #print(mean.shape)
        x = x - mean
        std = np.clip(np.std(x,axis=0),1e-15,1e15)
        x = x / std
    else:
        x = x - mean
        std = np.clip(std,1e-15,1e15)
        x = x / std
    return x, mean, std

def drop_high_nan_columns(data, threshold=0.33):
    nan_proportions = np.mean(np.isnan(data), axis=0)
    cols_to_keep = nan_proportions <= threshold
    
    # Finding the indices that were dropped
    dropped_indices = np.where(nan_proportions > threshold)[0].tolist()
    return data[:, cols_to_keep], dropped_indices,

def handle_outliers(x, numerical_columns, coefficient=2, mean=None, std_x=None):
    """ Addresses potential outliers in numerical columns by comparing z-scores """
    cp_x = x[:,numerical_columns].copy()
    cp_x[cp_x == -999] = np.nan
    
    if mean is None or std_x is None:
        mean = np.nanmean(cp_x, axis=0)
        std_x = np.nanstd(cp_x, axis=0)
    else:
        pass


    if coefficient != -1:
        lower_bound = mean - coefficient * std_x
        upper_bound = mean + coefficient * std_x
        cp_x_up=upper_bound-cp_x
        cp_x_lo=lower_bound-cp_x
        cp_x[(cp_x_up<0) & (cp_x_lo>0)]=np.nan

    returned_data=x.copy()
    returned_data[:,numerical_columns]=cp_x
    return returned_data, mean, std_x


def handle_missing_values_np(data):
    """
    Handles missing values using a given strategy.
    
    Args:
    - data (np.ndarray): The array with missing values.
    
    Returns:
    - np.ndarray: The array with missing values handled.
    """
    most_frequent = []
    for column in data.T:
        unique_vals, counts = np.unique(column[~np.isnan(column)], return_counts=True)
        most_frequent.append(unique_vals[np.argmax(counts)])
    most_frequent = np.array(most_frequent)
    return np.where(np.isnan(data), most_frequent, data)

def select_categorical(x_train):
    '''
    Checks the count of non-nan unique values and determines which columns are going to be taken as
    categorical.
    '''
    unique_values_count = np.array([len(np.unique(column[~np.isnan(column)])) for column in x_train.T])
    categorical_columns_indices = np.where(unique_values_count <= 10)[0]
    numerical_columns_indices = np.where(unique_values_count > 10)[0]

    return categorical_columns_indices,numerical_columns_indices

def one_hot_encode(data_train, data_test, categorical_columns_indices):
    """
    One-hot encodes columns with 10 or fewer unique non-NaN values.

    Args:
    - data_train (np.array): Training data to be one-hot encoded.
    - data_test (np.array): Testing data to be one-hot encoded.
    - unique_values_count (np.array): Array containing the count of unique values for each column.

    Returns:
    - np.array: One-hot encoded training data.
    - np.array: One-hot encoded testing data.
    """
    
    # Identify which columns need one-hot encoding
    
    # Apply one-hot encoding
    one_hot_encoded_data_train = []
    one_hot_encoded_data_test = []

    for idx in categorical_columns_indices:
        # Extract unique non-NaN values from the training data
        unique_vals = np.unique(data_train[:, idx][~np.isnan(data_train[:, idx])])
        for val in unique_vals:
            one_hot_encoded_data_train.append((data_train[:, idx] == val).astype(int))
            # For the test data, check if the value exists; if not, create a column of zeros
            if val in data_test[:, idx]:
                one_hot_encoded_data_test.append((data_test[:, idx] == val).astype(int))
            else:
                one_hot_encoded_data_test.append(np.zeros(data_test.shape[0]))

    # Stack them together for the final one-hot encoded matrices
    one_hot_encoded_data_train = np.stack(one_hot_encoded_data_train, axis=1)
    one_hot_encoded_data_test = np.stack(one_hot_encoded_data_test, axis=1)

    # Remove the original categorical columns and append the one-hot encoded columns
    data_train = np.delete(data_train, categorical_columns_indices, axis=1)
    data_train = np.hstack((data_train, one_hot_encoded_data_train))

    data_test = np.delete(data_test, categorical_columns_indices, axis=1)
    data_test = np.hstack((data_test, one_hot_encoded_data_test))
    
    return data_train, data_test


# Feature Selection
def remove_highly_correlated_features(X_train, X_test, threshold=0.95):
    """
    Remove one of the columns that has a correlation higher than the specified threshold.
        X_train: Features of the training data (N1, D).
        X_test: Features of the testing data (N2, D).
        threshold: Correlation threshold for feature removal (default is 0.95).
    Returns:
        reduced_X_train: The resulting training data features with correlated features removed.
        reduced_X_test: The resulting testing data features with correlated features removed.
        remaining_col_names: The names of the columns that remain after removing correlated features.
    """
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    high_correlation_indices = np.where(np.triu(np.abs(corr_matrix) > threshold, k=1))
    columns_to_remove = set()

    for i, j in zip(*high_correlation_indices):
        col_i_corr_sum = np.sum(np.abs(corr_matrix[i]))
        col_j_corr_sum = np.sum(np.abs(corr_matrix[j]))
        if col_i_corr_sum > col_j_corr_sum:
            columns_to_remove.add(j)
        else:
            columns_to_remove.add(i)

    reduced_X_train = np.delete(X_train, list(columns_to_remove), axis=1)
    reduced_X_test = np.delete(X_test, list(columns_to_remove), axis=1) if X_test is not None else None

    return reduced_X_train, reduced_X_test


def apply_preprocessing(x_train, x_test, coefficient=2):
    
    categorical_columns_indices, numerical_columns_indices = select_categorical(x_train)
    
    x_train, mean, std_x = handle_outliers(x_train, numerical_columns_indices, coefficient)
    x_test, _, _ = handle_outliers(x_test, numerical_columns_indices, coefficient, mean, std_x)
    #print("imhere")

    x_train, dropped_train_cols = drop_high_nan_columns(x_train)
    x_test = np.delete(x_test, dropped_train_cols, axis=1)

    x_train = handle_missing_values_np(x_train)
    x_test = handle_missing_values_np(x_test)
    
    categorical_columns_indices, numerical_columns_indices = select_categorical(x_train)
    x_train, x_test = one_hot_encode(x_train, x_test, categorical_columns_indices)

    x_train, x_test = remove_highly_correlated_features(x_train, x_test)

    return x_train, x_test, numerical_columns_indices
