# House Price and Category Prediction with Multi-Task Deep Learning

## Overview

This project focuses on predicting house prices and categorizing houses based on various features. We begin by addressing data inconsistencies, missing values, and handling categorical variables to prepare a clean dataset for analysis. The preprocessing pipeline involves dropping irrelevant columns, creating a 'House Category' target variable, handling missing values, and encoding categorical features.

## Data Preprocessing

1. **Drop Irrelevant Columns:** We start by dropping the 'Id' column, which serves as a unique identifier and does not contribute to the analysis.

2. **House Categorization:** We categorize houses based on architectural style, building type, and construction year, creating a 'House Category' target variable.

3. **One-Hot Encoding:** Categorical target variables are one-hot encoded, generating binary columns representing different house categories.

4. **Missing Values Handling:** Columns with missing values exceeding 50% are dropped to ensure data integrity.

5. **Feature Transformation:** Numerical features are imputed with the mean and scaled using StandardScaler. High cardinality categorical features are imputed with the most frequent value and one-hot encoded, while ordinal categorical features are similarly imputed and encoded.

6. **Column Transformation:** The ColumnTransformer applies distinct transformations to numerical, high cardinality categorical, and ordinal categorical features.

## Multi-Task Deep Learning

We employ a multi-task deep learning model using PyTorch and PyTorch Lightning to simultaneously predict house prices and categories.

1. **Custom Dataset:** We create a custom dataset class to convert the processed features and targets into PyTorch tensors.

2. **Shared Bottom Model:** A fully connected neural network with one hidden layer serves as the shared bottom model. It applies linear transformations and the specified activation function to the input.

3. **Multi-Task Model:** Derived from PyTorch Lightning's LightningModule, this model calculates both regression (RMSE) and classification (cross-entropy) losses during training.

4. **Training and Optimization:** The model is trained using the Adam optimizer with a learning rate of 0.1. Hyperparameter optimization is performed using Optuna to achieve the best model performance.

## Hyperparameter Optimization

1. **Optuna Study:** We create an Optuna study object to search for the best hyperparameters.

2. **Nested Loops:** We iterate over activation functions and optimizers, creating and training models for each combination.

3. **Validation Loss Comparison:** The validation loss is compared for each combination, and the best hyperparameters are retained.

## Performance Enhancement

Through hyperparameter optimization, the best validation loss was reduced significantly, resulting in a more accurate model for predicting house prices and categories. The final validation loss demonstrates improved model performance and better alignment between predicted and actual values.

## Conclusion

This project showcases the power of deep learning and hyperparameter optimization in enhancing predictive models. The multi-task deep learning model, coupled with Optuna's optimization algorithms, provides accurate predictions for both house prices and categories. The iterative tuning process demonstrates the importance of hyperparameter tuning in maximizing model potential and improving generalization on unseen data.
