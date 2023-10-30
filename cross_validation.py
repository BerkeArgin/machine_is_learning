import numpy as np
from preprocess import *
from itertools import product


def stratified_K_fold(y, k=5, seed=42):
    np.random.seed(seed)
    no_of_test_labels = np.floor(y.shape[0] / k)

    unique_labels = np.unique(y)
    label_ratio = {}
    label_indexes = {}
    label_count_p_fold = {}

    picked_count = 0
    for i, label in enumerate(unique_labels):
        label_ratio[label] = y[y == label].shape[0] / y.shape[0]
        label_indexes[label] = np.random.permutation(np.argwhere(y == label))
        if i < len(unique_labels) - 1:
            label_count_p_fold[label] = np.floor(no_of_test_labels * label_ratio[label])
            picked_count += label_count_p_fold[label]
        else:
            label_count_p_fold[label] = no_of_test_labels - picked_count

    folds = []
    for i in range(k):
        index_arrs = []
        for label in unique_labels:
            label_to_take = int(label_count_p_fold[label])
            index_arrs.append(
                label_indexes[label][
                    i * label_to_take : (i + 1) * label_to_take
                ].flatten()
            )

        test_fold = np.random.permutation(np.concatenate(index_arrs))
        train_fold = np.random.permutation(
            np.setdiff1d(np.indices(y.shape), test_fold, assume_unique=True)
        )
        folds.append((train_fold, test_fold))

    return folds


def calculate_metrics(y_pred, y_true):
    accuracy = y_pred[y_pred == y_true].shape[0] / y_pred.shape[0]
    precision = {}
    recall = {}
    for label in np.unique(y_true):
        try:
            precision[label] = (
                y_pred[(y_pred == label) & (y_true == label)].shape[0]
                / y_pred[y_pred == label].shape[0]
            )
        except:
            continue
        try:
            recall[label] = (
                y_pred[(y_pred == label) & (y_true == label)].shape[0]
                / y_true[y_true == label].shape[0]
            )
        except:
            continue

    f1_scores = {}
    for label in precision.keys():
        if precision[label] + recall[label] != 0:
            f1_scores[label] = (
                2
                * (precision[label] * recall[label])
                / (precision[label] + recall[label])
            )
        else:
            f1_scores[label] = 0.0

    return accuracy, precision, recall, f1_scores


def calculate_confusion_mat(y_pred, y_true):
    unique_labels = np.unique(y_true)
    conf_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    # print(conf_matrix.shape)
    for i, pred_label in enumerate(unique_labels):
        for j, true_label in enumerate(unique_labels):
            conf_matrix[i, j] = y_pred[
                (y_pred == pred_label) & (y_true == true_label)
            ].shape[0]
    return conf_matrix


def cross_validate(
    X_train,
    y_train,
    k_folds,
    train_func,
    loss_function,
    predict_function,
    *args,
    **kwargs,
):
    results = []
    for i, (train_fold, val_fold) in enumerate(k_folds):
        results_dict = {}
        # print("Fold",i+1)
        x_train_fold, x_val_fold = (X_train[train_fold], X_train[val_fold])
        y_train_fold = y_train[train_fold]
        y_val_fold = y_train[val_fold]

        x_train_fold, mean, std = standardize(x_train_fold)
        x_val_fold, _, _ = standardize(x_val_fold, mean, std)

        initial_w = np.zeros((x_train_fold.shape[1],))

        X_undersampled, y_undersampled = undersample(X_train, y_train)

        try:
            w, loss = train_func(
                y_undersampled, X_undersampled, initial_w=initial_w, *args, **kwargs
            )
        except:
            w, loss = train_func(y_undersampled, X_undersampled, *args, **kwargs)

        y_pred, scores = predict_function(X_undersampled, w, threshold=0.5)
        tr_accuracy, tr_precision, tr_recall, tr_f1_scores = calculate_metrics(
            y_undersampled, y_pred
        )

        y_pred, scores = predict_function(x_val_fold, w, threshold=0.5)
        vl_accuracy, vl_precision, vl_recall, vl_f1_scores = calculate_metrics(
            y_val_fold, y_pred
        )

        results_dict["name"] = train_func.__name__
        results_dict["loss_type"] = loss_function.__name__
        results_dict["predict_type"] = predict_function.__name__

        results_dict.update(kwargs)
        results_dict["w"] = w
        results_dict["fold"] = i + 1

        results_dict["tr_loss"] = loss
        results_dict["tr_accuracy"] = tr_accuracy
        results_dict["tr_precision"] = tr_precision
        results_dict["tr_recall"] = tr_recall
        results_dict["tr_f1_scores"] = tr_f1_scores

        results_dict["vl_loss"] = loss_function(y_val_fold, x_val_fold, w)
        results_dict["vl_accuracy"] = vl_accuracy
        results_dict["vl_precision"] = vl_precision
        results_dict["vl_recall"] = vl_recall
        results_dict["vl_f1_scores"] = vl_f1_scores

        results_dict["conf_matrix"] = calculate_confusion_mat(y_pred, y_val_fold)

        results.append(results_dict)

    return results


def apply_cross_val(
    functions, grid_search_dict, function_kwargs, X_train, y_train, k_folds
):
    all_results = []
    for i, function in enumerate(functions):
        hyperparameters = function_kwargs[function[0].__name__]

        print(
            f"{i+1}/{len(functions)} STARTING: ({function[0].__name__}, {function[1].__name__} , {function[2].__name__})"
        )

        if len(hyperparameters) == 0:
            results = cross_validate(X_train, y_train, k_folds, *function)
            all_results.extend(results)
            continue

        param_combinations = product(
            *[grid_search_dict[param] for param in hyperparameters]
        )

        for param in param_combinations:
            kwarg = dict(zip(hyperparameters, list(param)))
            results = cross_validate(X_train, y_train, k_folds, *function, **kwarg)
            print(kwarg)
            all_results.extend(results)

        print(
            f"{i+1}/{len(functions)} DONE: ({function[0].__name__}, {function[1].__name__} , {function[2].__name__})"
        )
        return all_results
