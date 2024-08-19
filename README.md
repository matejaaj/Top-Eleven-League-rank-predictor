# Top Eleven: League Rank Prediction

## Academic Context

This project was developed as part of the coursework for "Numerical Algorithms and Numerical Software," a subject at the "Faculty of Technical Sciences." The task was originally part of the Nordeus Data Science Challenge that I took part in after JobFair 2023.


## Project Overview

This project aims to predict the end-of-season positions of teams in the Top Eleven game, using a dataset provided by Nordeus for their Data Science Challenge. It involves comparing various regression and classification models.

## Problem Description

The challenge is in accurately modeling the complex factors that influence team success using relevant features.

## Data Overview

The dataset includes over 50,000 rows and 23 columns, focusing on team attributes with 'league rank' as the target variable.

## Feature Selection

Features were identified using Random Forest Regressor's feature_importances and PCA analysis. Additionaly features were normalized for contextual relevance within one league (ranking within one league).

## Models

### Regression Models

- **MLPRegressor, Random Forest Regressor, Ordinary Least Squares Linear Regression**
- Evaluation based on Adjusted R-Squared and MAE.

### Classification Models

- **MLPClassifier, Random Forest Classifier**
- Evaluated on accuracy.

## Results

Regression and classification models did not perform as expected, indicating the need for advanced models for ranking within classes.

![](https://github.com/matejaaj/Top-Eleven-League-rank-predictor/blob/main/results/regression/results_regression.png)
![](https://github.com/matejaaj/Top-Eleven-League-rank-predictor/blob/main/results/classification/results_classification.png)

## Conclusion

The project highlights the need for classification models that can handle rankings more effectively.
