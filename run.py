import numpy as np
from helpers import *
from implementations import *
from preprocess import *
from cross_validation import *


variables_set = {
    "PAQ650",
    "_RACEGR3",
    "DRNK3GE5",
    "_FRTRESP",
    "CHILDREN",
    "VEGETAB1",
    "PAQ670",
    "_SMOKER3",
    "_RFBMI5",
    "TOLDHI2",
    "_RFHLTH",
    "_PAINDX2",
    "INCOME2",
    "_HISPANC",
    "FVBEANS",
    "_DRNKWEK",
    "PA1MIN_",
    "FRUIT1",
    "_HCVU651",
    "_AGE80",
    "HTM4",
    "_CHOLCHK",
    "FVGREEN",
    "VEGEDA1_",
    "ALCDAY5",
    "_RFPAVIG",
    "_RFSMOK3",
    "_RACE_G1",
    "BEANDAY_",
    "_CASTHM1",
    "ORNGDAY_",
    "SMOKDAY2",
    "_CHLDCNT",
    "_METVL11",
    "PAQ665",
    "_AGEG5YR",
    "MRACORG1",
    "ORACE3",
    "SEX",
    "HAVARTH3",
    "HISPANC3",
    "_PAQ6C",
    "DROCDY3_",
    "AGE",
    "HLTHPLN1",
    "_MICHD",
    "FTJUDA1_",
    "_VEGRESP",
    "_RFDRHV5",
    "FRUTDA1_",
    "_VEGESUM",
    "WEIGHT2",
    "_MISFRTN",
    "CVDCRHD4",
    "BPHIGH4",
    "_DRDXAR1",
    "XPA1MIN_",
    "_PASTRNG",
    "_RACEG21",
    "_IMPAGE",
    "_PRACE1",
    "_INCOMG",
    "AVEDRNK",
    "_METVL21",
    "_MISVEGN",
    "_RFCHOL",
    "PAQ655",
    "HEIGHT3",
    "_PAREC1",
    "_RFBING5",
    "_PACAT1",
    "_FRUTVEG",
    "_RFHYPE5",
    "DRNKANY5",
    "_MRACE1",
    "GRENDAY_",
    "PAQ660",
    "GENHLTH",
    "_BMI5",
    "FVORANG",
    "_EDUCAG",
    "_BMI5CAT",
    "_LTASTH1",
    "MRACE1",
    "_ASTHMS1",
    "_FRUTSUM",
    "ASTHMA3",
    "BLOODCHO",
    "SMOKE100",
    "_AGE_G",
    "EDUCA",
    "WTKG3",
    "ASTHNOW",
    "CVDINFR4",
    "HTIN4",
    "_RACE",
    "FRUITJU1",
    "MRACASC1",
    "_AGE65YR",
    "CHOLCHK",
}


def undersample(X, y, seed=42):
    np.random.seed(seed)

    minority_count = y[y == 1].shape[0]

    undersampled_majority_indices = np.random.choice(
        np.where(y == 0)[0], minority_count, replace=False
    )
    undersampled_indices = np.random.permutation(
        np.concatenate([undersampled_majority_indices, np.where(y == 1)[0]])
    )

    X_undersampled = X[undersampled_indices]
    y_undersampled = y[undersampled_indices]

    return X_undersampled, y_undersampled


data_path = "./data/dataset_to_release"
(
    x_train,
    x_test,
    y_train,
    train_ids,
    test_ids,
    col_names_train,
    col_names_test,
    final_columns,
) = load_csv_data(data_path, selected_cols=variables_set)

y_train[y_train == -1] = 0
X_train, X_test, numerical_columns_indices = apply_preprocessing(x_train, x_test)

X_train, mean, std = standardize(X_train)
X_test, _, _ = standardize(X_test, mean, std)

X_undersampled, y_undersampled = undersample(X_train, y_train)


initial_w = np.zeros((X_train.shape[1],))

w, loss = ridge_regression(y_undersampled, X_undersampled, lambda_=0.01)

y_pred, scores = ridge_predict(X_test, w, threshold=0.5)

y_pred[y_pred == 0] = -1
create_csv_submission(test_ids, y_pred, "submission.csv")


###for cross validation
"""
functions=[
    (reg_logistic_regression,calculate_logistic_loss,logit_predict),
    #(least_squares,calculate_mse_loss,least_sq_predict),
    (logistic_regression,calculate_logistic_loss,logit_predict),
    (ridge_regression,calculate_mse_loss,ridge_predict),
    (ridge_regression,calculate_mse_loss,logit_predict),
    (mean_squared_error_sgd,calculate_mse_loss,least_sq_predict),
    (mean_squared_error_gd,calculate_mse_loss,least_sq_predict)
    ]

    grid_search_dict={
        "lambda_" : [1,0.5,0.1,0.01],
        "gamma" : [0.01,0.001,0.0001],
        "max_iters" : [100,250,500]
    }

    function_kwargs= {
    "ridge_regression_gd" : ["lambda_","gamma","max_iters"],
    "reg_logistic_regression" : ["lambda_","gamma","max_iters"],
    "logistic_regression" : ["gamma","max_iters"],
    "ridge_regression" : ["lambda_"],
    "least_squares" : [],
    "mean_squared_error_sgd" : ["gamma","max_iters"],
    "mean_squared_error_gd" : ["gamma","max_iters"] 
    }
k_folds=stratified_K_fold(y_train,seed=42)
cross_val_res = apply_cross_val(functions, grid_search_dict, function_kwargs, X_train,y_train,k_folds )
"""
