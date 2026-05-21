# README.md

# California Housing ML Evaluation

A machine learning evaluation project using the California Housing dataset from scikit-learn. The project explores preprocessing pipelines, feature engineering, regression model comparison, and different validation strategies using Python and scikit-learn.

## Overview

This project compares several regression approaches for predicting California housing prices:

* Ridge Regression
* Lasso Regression
* K-Nearest Neighbors (KNN)

The workflow includes:

* Feature engineering
* Standardized preprocessing pipelines
* Multiple train/validation/test split strategies
* TimeSeriesSplit cross-validation
* Regression metric evaluation (R², MAE, MSE)

## Technologies Used

* Python
* pandas
* scikit-learn

## Key Concepts Explored

* Regression modeling
* Feature engineering
* Preprocessing pipelines
* Cross-validation
* Model evaluation
* Train/validation/test workflows
* Comparative model analysis

## Dataset

The project uses the California Housing dataset available through:

```python
sklearn.datasets.fetch_california_housing
```

## Example Workflow

* Load and preprocess housing data
* Engineer additional features
* Train Ridge, Lasso, and KNN models
* Compare performance across multiple validation schemes
* Export evaluation results to CSV

## Goal

The goal of this project was to explore practical ML workflows, preprocessing techniques, model evaluation strategies, and how different regression models behave under different validation approaches.
