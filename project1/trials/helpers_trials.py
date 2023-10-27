"""Some helper functions for project 1."""
import csv
import numpy as np
import os

def calculate_mse_loss(y, tx, w):
    """
    Calculate Mean Absolute Error Loss
    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
       The Mean Squared Error Loss between observed and predicted outputs

    """
    y_pred = tx.dot(w)
    return np.mean((y - y_pred) ** 2) / 2


def calculate_gradient(y, tx, w):
    """
    This function calculates the gradient at w and return gradient and error values.
    Args:
        tx: A numpy array of shape (N,) containing the observed outputs.
        error:

    Returns:
       The gradient of least squares (shape (D,)) and the error between observed and predicted outputs (shape (N,))
    """
    error = y - tx.dot(w)
    return -tx.T.dot(error) / error.size


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Args:
        y:  The output desired values.
        tx: The input data.
        batch_size: Size of each mini-batch.
        num_batches: The number of mini-batches to produce.
        shuffle: If True, the data is shuffled; otherwise, data is taken sequentially.

    Returns:
       Containing the output labels and input data for a mini-batch.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
            data_size - max_batches * batch_size
    )
    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            # Without an offset, batches might start at indices: [0, 32, 64, 96] (assume remainder=4)
            # With a random offset, batches would start at: [4, 36, 68, 100]
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size
    for start in idxs:
        start_index = start
        end_index = (
                start_index + batch_size
        )
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(x):
    """
    Applies the sigmoid function to a given input.
    Args:
        x: Given input.
    Returns:
        The value of sigmoid function.
    """
    return 1. / (1. + np.exp(-x))

def calculate_logistic_loss(y, tx, w):
    """
    Compute the negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
        Negative log likelihood loss.
    """
    t = np.dot(tx, w)
    loss = np.sum(np.log(1 + np.exp(t)) - y * t) / y.shape[0]
    return loss


def calculate_weighted_logistic_loss(y, tx, w, w1, w2):
    """
    Compute the weighted negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.
        w1: Weight for class 1 (minority class).
        w2: Weight for class 0 (majority class).

    Returns:
        Weighted negative log likelihood loss.
    """
    t = np.dot(tx, w)
    pred_probs = 1 / (1 + np.exp(-t))
    loss = -np.mean(w1 * y * np.log(pred_probs + 1e-15) + w2 * (1 - y) * np.log(1 - pred_probs + 1e-15))
    return loss

def calculate_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.

    Returns:
        gradient_vector: Gradient vector which has shape (D,)
    """
    predicted_probs = sigmoid(tx.dot(w))
    gradient_vector = tx.T.dot(predicted_probs - y)
    return gradient_vector

def calculate_logistic_regression_regularized(y, tx, w, lambda_):
    """
        Compute the gradient of the negative log likelihood for logistic regression with regularization term for l2".

        Args:
            y: A numpy array of shape (N,) containing the observed outputs.
            tx: A numpy array of shape (N, D) containing the feature matrix of the data.
            w: A numpy array of shape (D,) which is the weight vector.
            lambda_: Regularization parameter.
        Returns:
            gradient_vector_regularized: Gradient vector which has shape (D,)
        """
    predictions = sigmoid(tx.dot(w))
    gradient_vector_regularized = tx.T.dot(predictions - y) / y.shape[0] + 2 * lambda_ * w
    return gradient_vector_regularized


def calculate_weighted_logistic_loss(y, tx, w, w1, w2):
    """
    Compute the weighted negative log likelihood for logistic regression.

    Args:
        y: A numpy array of shape (N,) containing the observed outputs.
        tx: A numpy array of shape (N, D) containing the feature matrix of the data.
        w: A numpy array of shape (D,) which is the weight vector.
        w1: Weight for class 1 (minority class).
        w2: Weight for class 0 (majority class).

    Returns:
        Weighted negative log likelihood loss.
    """
    t = np.dot(tx, w)
    pred_probs = 1 / (1 + np.exp(-t))
    loss = -np.mean(w1 * y * np.log(pred_probs + 1e-15) + w2 * (1 - y) * np.log(1 - pred_probs + 1e-15))
    return loss


def load_csv_data(data_path, sub_sample=False, selected_cols=None):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    with open(os.path.join(data_path, "x_train.csv"), 'r') as f:
        header = f.readline().strip().split(',')

    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    col_names_train = np.genfromtxt(data_path + "/x_train.csv", delimiter=',', max_rows=1, dtype=str).tolist()
    col_names_test = np.genfromtxt(data_path + "/x_test.csv", delimiter=',', max_rows=1, dtype=str).tolist()
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # Select only the specified columns
    if selected_cols is not None:
        selected_indices=[]
        for col_name in selected_cols:
            if col_name in header:
                selected_indices.append(header.index(col_name) - 1)
        x_train = x_train[:, selected_indices]
        x_test = x_test[:,selected_indices]
    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids, col_names_train, col_names_test

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

###### WEIGHTED CALCULATIONS
def calculate_weighted_mse_loss(y, tx, w, weights):
    """Compute the weighted mean squared error loss."""
    e = y - tx.dot(w)
    return np.sum(weights * e**2) / (2 * len(y))

def calculate_weighted_gradient(y, tx, w, weights):
    """Compute the gradient for weighted MSE."""
    e = y - tx.dot(w)
    return -tx.T.dot(weights * e) / len(y)

def calculate_logistic_loss_weighted(y, tx, w, weights):
    """Compute the weighted logistic loss."""
    pred = tx.dot(w)
    log_likelihood = y * pred - np.log(1 + np.exp(pred))
    return -np.sum(weights * log_likelihood) / len(y)

def calculate_logistic_gradient_weighted(y, tx, w, weights):
    """Compute the gradient for weighted logistic loss."""
    pred = tx.dot(w)
    sigma = 1 / (1 + np.exp(-pred))
    return tx.T.dot(weights * (sigma - y)) / len(y)

def calculate_logistic_loss_regularized_weighted(y, tx, w, lambda_, weights):
    """Compute the weighted regularized logistic loss."""
    pred = tx.dot(w)
    log_likelihood = y * pred - np.log(1 + np.exp(pred))
    reg_term = lambda_ * np.linalg.norm(w, 2) ** 2
    return (-np.sum(weights * log_likelihood) + reg_term) / len(y)

def calculate_logistic_gradient_regularized_weighted(y, tx, w, lambda_, weights):
    """Compute the gradient for weighted regularized logistic loss."""
    pred = tx.dot(w)
    sigma = 1 / (1 + np.exp(-pred))
    reg_term_gradient = 2 * lambda_ * w
    return tx.T.dot(weights * (sigma - y)) / len(y) + reg_term_gradient

