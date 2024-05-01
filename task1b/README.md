# Advanced Feature Engineering for Linear Regression

## Overview
This project applies advanced feature engineering techniques to enhance the performance of linear regression models. By transforming input features into a higher-dimensional space that includes linear, quadratic, exponential, cosine, and constant transformations, the model can capture more complex patterns in the data.

## Data
The `train.csv` dataset comprises an ID, the target variable `y`, and five predictor variables `x1` to `x5`. The transformation expands these five predictors into 21 features to be used in the regression model, improving the capacity of the model to learn from complex relationships.

## Implementation
- **Feature Transformation**: Inputs are expanded to 21 features using linear, quadratic, exponential, and cosine transformations, plus a constant feature.
- **Ridge Regression with Cross-Validation**: The transformed data is scaled using `StandardScaler` and then fitted to a Ridge regression model. The model uses cross-validation (`RidgeCV`) to find the optimal regularization parameter.
- **Evaluation**: The script outputs the regression coefficients into `results.csv`, which can be used to assess the model's predictions against actual values.

## Code Structure
- **Data Loading**: The data is loaded from `train.csv`, separating features from the label.
- **Data Transformation**: The `transform_data` function is applied to the features to generate the expanded feature set.
- **Model Fitting**: A `RidgeCV` model is fitted to the standardized transformed data. The optimal regularization strength is selected automatically.
- **Output**: The fitted model's coefficients are saved to `results.csv`.

The script is designed to run end-to-end from data loading to output generation, ensuring that all steps are reproducible and well-documented.
