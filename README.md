# Austin Airbnb Price Prediction

Machine learning analysis of Austin Airbnb listings to predict nightly rental prices and compare the performance of multiple statistical and machine learning models.

## Project Overview

This project analyzes Austin Airbnb listing data and builds predictive models for nightly prices. The analysis includes data preprocessing, regression diagnostics, feature transformations, cross-validation, hyperparameter tuning, and out-of-sample model evaluation.

## Models Compared

- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Single Decision Tree
- Random Forest
- Gradient Boosting

## Methodology

The dataset was divided into an 80% training set and 20% test set.

Nightly price was log-transformed after residual diagnostics indicated unequal variance in the original price model.

Five-fold cross-validation was used to tune and compare all six models using the same folds.

Polynomial transformations were incorporated for predictors exhibiting nonlinear relationships.

## Results

| Model | CV RMSE | Test RMSE |
|---|---:|---:|
| OLS | 0.400 | 0.385 |
| Ridge | 0.401 | 0.386 |
| Lasso | 0.400 | 0.385 |
| Single Tree | 0.435 | 0.409 |
| Random Forest | 0.363 | 0.349 |
| Gradient Boosting | **0.351** | **0.342** |

Gradient boosting produced the lowest cross-validation and test RMSE, outperforming both the linear models and other tree-based approaches.

## Tools & Technologies

- R
- tidyverse
- caret
- recipes
- glmnet
- ranger
- gbm
- ggplot2

## Key Skills Demonstrated

- Predictive modeling
- Cross-validation
- Hyperparameter tuning
- Regression diagnostics
- Feature engineering
- Regularization
- Ensemble learning
- Model evaluation
- Data visualization
