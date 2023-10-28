from helpers import *
import numpy as np

def drop_high_nan_columns(data, threshold=0.33):
    nan_proportions = np.mean(np.isnan(data), axis=0)
    cols_to_keep = nan_proportions <= threshold
    
    # Finding the indices that were dropped
    dropped_indices = np.where(nan_proportions > threshold)[0].tolist()
    return data[:, cols_to_keep], dropped_indices, 

def undersample(x, y):
    """
    Perform undersampling on the majority class.
    
    Parameters:
    - x: Feature data (numpy array)
    - y: Target labels (numpy array)
    
    Returns:
    - Undersampled feature data and corresponding labels.
    """
    # Separate majority and minority class indices
    majority_indices = np.where(y == -1)[0]
    minority_indices = np.where(y == 1)[0]
    
    # Randomly select samples from majority class to match minority class count
    random_majority_indices = np.random.choice(majority_indices, len(minority_indices), replace=False)
    
    # Combine the down-sampled majority class indices with the original minority class indices
    undersampled_indices = np.concatenate([minority_indices, random_majority_indices])
    
    # Get the undersampled data and labels
    x_undersampled = x[undersampled_indices]
    y_undersampled = y[undersampled_indices]
    
    return x_undersampled, y_undersampled

def min_max_scaling_array(data_array, scaling_parameters=None):
    """
    Apply Min-Max Scaling to a numpy array.
    
    Parameters:
    - data_array: 2D numpy array where rows are data entries and columns represent features.
    - scaling_parameters: Optional precomputed scaling parameters (min and max values).
    
    Returns:
    - Numpy array of scaled data.
    - Scaling parameters used.
    """
    if scaling_parameters is None:
        scaling_parameters = {}
        
    num_cols = data_array.shape[1]
    
    for col_idx in range(num_cols):
        col_data = data_array[:, col_idx]
        if col_idx not in scaling_parameters:
            col_min = np.nanmin(col_data)
            col_max = np.nanmax(col_data)
            scaling_parameters[col_idx] = (col_min, col_max)
        else:
            col_min, col_max = scaling_parameters[col_idx]
        
        # Apply scaling
        # Avoid division by zero
        diff = col_max - col_min
        if diff == 0:
            continue
        else:
            col_data = (col_data - col_min) / diff

        data_array[:, col_idx] = col_data
            
    return data_array, scaling_parameters  # Ensure we're returning both the scaled array and the parameters


def apply_preprocessing(x_train, x_test, categorical_columns_indices, all_categories):
    """
    Apply the preprocessing functions on the given train and test datasets.
    """
    
    # 1. Impute missing values
    x_train_imputed, _ = impute_missing_values(x_train)
    x_test_imputed, _ = impute_missing_values(x_test)
    if np.isnan(x_train_imputed).any():
        print(f"NaNs detected in after imputing")


    # 2. One-hot encoding
    x_train_cat = x_train_imputed[:, categorical_columns_indices]
    x_test_cat = x_test_imputed[:, categorical_columns_indices]
    x_train_cat_encoded, x_test_cat_encoded = one_hot_encoding(x_train_cat, x_test_cat, categorical_columns_indices, all_categories)
    
    # Update x_train and x_test by replacing the original categorical columns with one-hot encoded columns
    x_train_imputed = np.hstack((np.delete(x_train_imputed, categorical_columns_indices, axis=1), x_train_cat_encoded))
    x_test_imputed = np.hstack((np.delete(x_test_imputed, categorical_columns_indices, axis=1), x_test_cat_encoded))
    if np.isnan(x_train_imputed).any():
            print(f"NaNs detected in after one hot ")

    # 3. Apply Min-Max Scaling to all columns
    x_train_imputed, scaling_params = min_max_scaling_array(x_train_imputed)
    x_test_imputed, _ = min_max_scaling_array(x_test_imputed, scaling_params)
    if np.isnan(x_train_imputed).any():
            print(f"NaNs detected in after minmax ")

    return x_train_imputed, x_test_imputed

# 1. Impute Missing Values

def impute_missing_values(data_array):
    """
    Impute the missing values in the data with the most frequent value of that column.
    
    Parameters:
    - data_array: 2D numpy array where rows are data entries and columns represent features.
    
    Returns:
    - Numpy array of imputed data.
    - Dictionary with imputation values for each column.
    """
    imputation_values = {}
    data_array = data_array.copy()
    num_cols = data_array.shape[1]
    
    for col_idx in range(num_cols):
        col_data = data_array[:, col_idx]
        # Check if there are any missing values in the column
        if np.isnan(col_data).any():
            # Get unique values and their counts, ignoring nan
            unique_vals, counts = np.unique(col_data[~np.isnan(col_data)], return_counts=True)
            most_frequent = unique_vals[np.argmax(counts)]
            imputation_values[col_idx] = most_frequent
            col_data[np.isnan(col_data)] = most_frequent
            data_array[:, col_idx] = col_data
            #print(f"Detected NaNs in column {col_idx}. Imputing with value {most_frequent}.")
            
    return data_array, imputation_values

def one_hot_encoding(data_train, data_test, categorical_columns_indices, all_categories):
    """
    Perform one-hot encoding on the specified categorical columns.
    
    Parameters:
    - data_train: Training data for a specific fold.
    - data_test: Test (or validation) data for a specific fold.
    - categorical_columns_indices: Indices of the columns that are categorical and need to be one-hot encoded.
    - all_categories: A list of numpy arrays, where each array contains all unique values for a categorical column 
                      across the entire dataset.
    
    Returns:
    - One-hot encoded training and test data for the specific fold.
    """
    one_hot_encoded_data_train = []
    one_hot_encoded_data_test = []

    for idx, column_categories in zip(categorical_columns_indices, all_categories):
        for val in column_categories:
            # Encode the value for the training data
            one_hot_encoded_data_train.append((data_train[:, idx] == val).astype(int))
            
            # Encode the value for the test data
            one_hot_encoded_data_test.append((data_test[:, idx] == val).astype(int))

    # Convert lists to numpy arrays for the final one-hot encoded matrices
    one_hot_encoded_data_train = np.stack(one_hot_encoded_data_train, axis=1)
    one_hot_encoded_data_test = np.stack(one_hot_encoded_data_test, axis=1)

    # Remove the original categorical columns and append the one-hot encoded columns
    data_train = np.delete(data_train, categorical_columns_indices, axis=1)
    data_train = np.hstack((data_train, one_hot_encoded_data_train))

    data_test = np.delete(data_test, categorical_columns_indices, axis=1)
    data_test = np.hstack((data_test, one_hot_encoded_data_test))
    
    return data_train, data_test




