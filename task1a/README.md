# Ridge Regression with Cross-Validation

## Overview
This Python project implements ridge regression combined with 10-fold cross-validation. It is designed to compute and report the Root Mean Squared Error (RMSE) for five different regularization parameters: 0.1, 1, 10, 100, and 200. The objective is to analyze how the regularization strength affects the predictive accuracy of the model.

## Data
The script operates on a dataset provided in `train.csv`, which includes labels and 13 features per datapoint. This dataset is used throughout the cross-validation process to train and test the ridge regression model.

## Implementation
- **fit(X, y, lam)**: Fits the ridge regression model using the provided data and regularization parameter.
- **calculate_RMSE(w, X, y)**: Calculates the RMSE between the predicted values and actual labels.
- **average_LR_RMSE(X, y, lambdas, n_folds)**: Conducts the cross-validation, calculating average RMSE for each lambda value.

The implementation leverages the scikit-learn library for the regression model and cross-validation functionality. The final RMSE results for each regularization strength are saved to `results.csv`.

## Code Structure
The main script orchestrates the loading of data, execution of the cross-validation routine, and saving of results. Each component function is designed to perform specific tasks that contribute to the overall experiment of assessing model performance across varying degrees of regularization.
