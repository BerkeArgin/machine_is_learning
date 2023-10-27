import numpy as np
from preprocess import *

def stratified_K_fold(y,k=5,seed=42):
    np.random.seed(seed)
    no_of_test_labels=np.floor(y.shape[0]/k)
    
    unique_labels=np.unique(y)
    label_ratio={};label_indexes={};label_count_p_fold={}

    picked_count=0
    for i,label in enumerate(unique_labels):
        label_ratio[label]=y[y==label].shape[0]/y.shape[0]
        label_indexes[label]=np.random.permutation(np.argwhere(y==label))
        if i<len(unique_labels)-1:
            label_count_p_fold[label]=np.floor(no_of_test_labels*label_ratio[label])
            picked_count+=label_count_p_fold[label]
        else:
            label_count_p_fold[label]=no_of_test_labels-picked_count

    folds=[]
    for i in range(k):
        index_arrs=[]
        for label in unique_labels:
            label_to_take=int(label_count_p_fold[label])
            index_arrs.append(label_indexes[label][i*label_to_take:(i+1)*label_to_take].flatten())
    
        test_fold=np.random.permutation(np.concatenate(index_arrs))
        train_fold=np.random.permutation(np.setdiff1d(np.indices(y.shape),test_fold,assume_unique=True))
        folds.append((train_fold,test_fold))
    
    return folds

def prepare_data_for_fold(y, x, k_indices, k):
    """
    Prepare the data for a given fold index.

    Args:
        y: Vector of labels
        x: Feature matrix
        k_indices: 2D array returned by build_k_indices()
        k: scalar, the k-th fold
        degree: Degree for polynomial expansion
        high_skewness_cols: Columns that need log transform
        high_residual_indices: Columns that need polynomial expansion

    Returns:
        train_x, train_y, test_x, test_y: Prepared data for the fold
    """
    
    test_x, test_y = x[k_indices[k]], y[k_indices[k]]
    train_x, train_y = (
        x[k_indices[(np.arange(len(k_indices)) != k)].reshape(-1)],
        y[k_indices[(np.arange(len(k_indices)) != k)].reshape(-1)],
    )
    

    
    #print(f"Shape of train_x before preprocessing: {train_x.shape}")
    #print(f"Shape of test_x before preprocessing: {test_x.shape}")
    
    #train_x, train_y = oversample_minority(train_x, train_y)
    train_x, test_x = apply_preprocessing(train_x, test_x)

    #print(f"Shape of train_x after preprocessing: {train_x.shape}")
    #print(f"Shape of test_x after preprocessing: {test_x.shape}")
    
    return train_x, train_y, test_x, test_y