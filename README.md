# EPFL Machine Learning Behavioral Risk Factor Surveillance System Project 1

In this project, we have implemented six machine learning methods to make predictions for the [Machine Learning Behavioral Risk Factor Surveillance System Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/leaderboards).

## Running the Model

To obtain the results that we submitted, follow these steps:

1. Create a folder called `data/dataset_to_release` in the project directory.
2. Place the `test.csv` and `train.csv` files under the created that folder in this directory.
3. Run the following command:
   ```shell
   python run.py
   ```
4. After executing `run.py`, the predictions for the test dataset will be saved as `submission.csv` in this directory.

The functions that we have applied for this project are located in implementations.py file. In addition to those six functions, the required functions for calculating the loss, batch iterations etc. are in that file. While preprocessing functions are in preprocess.py, the stratified K-fold function which we have used for cross validation and other required functions for cross validation are stored in cross_validation.py.

A commented section in the run.py file provides an example of how to use the cross_validation.py module.