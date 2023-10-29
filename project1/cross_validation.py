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
