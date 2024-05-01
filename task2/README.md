# Power Price Prediction

## Project Overview
This project, part of a machine learning course, aims to predict electricity prices in Switzerland using data from various countries and different seasons. It tackles common machine learning challenges such as handling missing data and dataset noise.

## Dataset
- **Training Dataset (`train.csv`)**: Contains seasonal data and electricity prices from multiple countries, with Switzerland's prices as the target variable.
- **Test Dataset (`test.csv`)**: Similar structure to the training set but without the target variable, used for predictions.
- **Sample Submission (`sample.csv`)**: Shows the expected format for submissions.

## Methodology
### Data Preprocessing
The initial phase involves loading data and addressing missing values to ensure robust model training. Using pandas for data manipulation, missing entries are imputed using sklearn's `SimpleImputer`, which replaces them with the mean of each column. This approach maintains the integrity and distribution of the dataset.

### Model Development and Selection
Kernel Ridge Regression is chosen for its effectiveness in capturing nonlinear relationships and its flexibility through the use of various kernels such as linear, polynomial, and RBF. This method is particularly suitable for regression tasks where prediction intervals are narrow, like price forecasting.

### Cross-validation and Hyperparameter Tuning
Model performance is rigorously evaluated using KFold cross-validation, ensuring that the model generalizes well across different subsets of data. This step involves testing various combinations of kernels and regularization parameters to identify the best model settings that minimize prediction errors.

### Final Model Training
Once the optimal parameters are identified, the final model is trained on the entire dataset. This model is then used to generate predictions for the test dataset, adhering to the submission format required by the course.

