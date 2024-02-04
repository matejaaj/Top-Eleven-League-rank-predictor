import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def getPcaModel(df, n_components=2, random_state=42):
    """
    Applies PCA on the given DataFrame after dropping specified columns and scaling.

    Parameters:
    - df: pandas.DataFrame, the input dataframe.
    - columns_to_drop: list of str, columns to drop from the dataframe.
    - n_components: int, number of principal components to keep..
    - random_state: int, random state for PCA.

    Returns:
    - pca_model: PCA, the PCA model after fitting.
    - principal_components: numpy.ndarray, the transformed data in the principal component space.
    """
    x = df.drop(columns=['league_rank', 'league_id'])
    x_scaled = scale(x)
    pca_model = PCA(n_components=n_components, random_state=random_state)
    principal_components = pca_model.fit_transform(x_scaled)
    
    return pca_model, principal_components

def getRandomForestClassifier(X_train, y_train):
    """
    Trains a Random Forest Classifier model on the provided training data.

    Parameters:
    - X_train: pandas.DataFrame or numpy.ndarray, feature matrix for training the model.
    - y_train: pandas.Series or numpy.ndarray, target values for training the model.

    Returns:
    - model: RandomForestClassifier, the trained Random Forest Classifier model.

    The model is configured with specific hyperparameters including number of estimators,
    maximum depth, minimum samples per leaf, minimum samples for a split, and maximum
    features to consider for the best split.
    """
    model = RandomForestClassifier(n_estimators=300, random_state=42,
                                   max_depth=10, bootstrap=True, min_samples_leaf=1,
                                   min_samples_split=10, max_features='sqrt')
    model.fit(X_train, y_train)
    return model


def getMLPClassifier(X_train, y_train):
    """
    Trains a Multilayer Perceptron (MLP) Classifier model on the provided training data.
    The input features are scaled before training to improve model performance.

    Parameters:
    - X_train: pandas.DataFrame or numpy.ndarray, feature matrix for training the model.
    - y_train: pandas.Series or numpy.ndarray, target values for training the model.

    Returns:
    - model: MLPClassifier, the trained MLP Classifier model.

    The MLP model is configured with a two-layer architecture, each layer having 50 neurons,
    and uses the ReLU activation function. The solver for weight optimization is 'adam',
    with a constant learning rate and a regularization term (alpha) set to improve
    generalization.
    """
    model = MLPClassifier(hidden_layer_sizes=(50,50),
                          activation='relu',         
                          solver='adam',
                          max_iter=1000,
                          learning_rate='constant',
                          alpha=0.05,
                          early_stopping=True,
                          random_state=42)

    model.fit(scale(X_train), y_train)
    return model



def getMLPRegressorModel(X_train, y_train):
    """
    Trains a Multilayer Perceptron (MLP) Regressor model with a specified architecture
    and hyperparameters on the provided training dataset.

    Parameters:
    - X_train: array-like or pandas.DataFrame, feature matrix for training the MLP model.
    - y_train: array-like or pandas.Series, target regression values for training.

    Returns:
    - model: MLPRegressor, the trained MLP Regressor model.

    The MLP Regressor is configured with two hidden layers of 100 and 50 neurons respectively,
    uses a maximum of 500 iterations for training, and a learning rate initialization of 0.001.
    The random state is fixed to ensure reproducibility of results.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42
    )
    model.fit(scale(X_train), y_train)
    return model



def getRandomForestModel(X_train, y_train):
    """
    Trains a Random Forest Regressor model with specified hyperparameters on the
    provided training dataset.

    Parameters:
    - X_train: array-like or pandas.DataFrame, feature matrix for training the Random Forest model.
    - y_train: array-like or pandas.Series, target regression values for training.

    Returns:
    - model: RandomForestRegressor, the trained Random Forest Regressor model.

    The model is configured with 300 trees, a maximum depth of 10, and controls over
    the minimum samples per leaf and split. Bootstrap samples are used. The configuration
    is designed to balance model complexity and overfitting, with a random state for
    reproducibility.
    """
    model = RandomForestRegressor(n_estimators=300, random_state=42,
                                  max_depth=10, bootstrap=True, min_samples_leaf=4,
                                  min_samples_split=2)
    model.fit(X_train, y_train)
    return model


def getLinearRegressionModel(X_train, y_train):
    """
    Fits an Ordinary Least Squares (OLS) Linear Regression model using the provided
    training data. Adds a constant term to the model to account for the intercept.

    Parameters:
    - X_train: array-like or pandas.DataFrame, feature matrix for training the OLS model.
    - y_train: array-like or pandas.Series, target regression values for training.

    Returns:
    - model: RegressionResultsWrapper, the fitted OLS regression model.

    This function uses the statsmodels API to fit the model, which provides detailed
    statistics about the model's performance. The constant term is added to include
    an intercept in the regression model.
    """
    x_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, x_with_const).fit()
    return model


def getFeatureImportancesPca(model):
    """
    Visualizes the feature importances of a trained model, assuming the model provides
    a `feature_importances_` attribute. This is particularly useful for models trained
    with PCA components to understand the relative importance of each principal component.

    Parameters:
    - model: A trained model instance with a `feature_importances_` attribute.

    Returns:
    - None. The function plots the feature importances using matplotlib.

    The importances are displayed in a bar chart, sorted from most to least important.
    The plot is useful for interpreting which components contribute most to the model's
    predictions.
    """
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("PCA Component Importances")
    plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align="center")
    plt.xticks(range(len(feature_importances)), [f"PC{i}" for i in sorted_idx], rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.show()



def getFeatureImportances(model, columns):
    """
    Visualizes the feature importances of a trained model based on the `feature_importances_`
    attribute. This function enhances interpretability by using actual feature names instead
    of PCA components.

    Parameters:
    - model: A trained model instance with a `feature_importances_` attribute.
    - columns: list of str, the names of the features corresponding to the model's feature importances.

    Returns:
    - None. The function plots the sorted feature importances using matplotlib.

    A bar chart displays the importances, with features labeled according to the provided
    column names, sorted from most to least important. This visualization aids in understanding
    which features are most influential in the model's decision-making process.
    """
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(columns)), feature_importances[sorted_idx], align="center")
    plt.xticks(range(len(columns)), [columns[i] for i in sorted_idx], rotation=90)
    plt.xlim([-1, len(columns)])
    plt.show()


def league_test_split(df, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and test sets based on unique league identifiers,
    ensuring that data from the same league is not mixed between the train and test sets.
    This means that every team from one leauge is in train/test set.
    This is used so that distribution of predictions is propperly evaluated.

    Parameters:
    - df: pandas.DataFrame, the dataset containing league identifiers and features.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train, X_test: Features for the training and test sets.
    - y_train, y_test: Target values for the training and test sets.

    This function is particularly useful for sports analytics and other scenarios where
    data leakage between similar groups (e.g., leagues) must be prevented.
    """
    leagues = df['league_id'].unique()
    train_leagues, test_leagues = train_test_split(leagues, test_size=test_size, random_state=random_state)

    X_train = df[df['league_id'].isin(train_leagues)].drop(['league_rank', 'league_id'], axis=1)
    y_train = df[df['league_id'].isin(train_leagues)]['league_rank']
    X_test = df[df['league_id'].isin(test_leagues)].drop(['league_rank', 'league_id'], axis=1)
    y_test = df[df['league_id'].isin(test_leagues)]['league_rank']
    return X_train, X_test, y_train, y_test


def scale(x):
    """
    Scales the input features using StandardScaler, which standardizes features by removing
    the mean and scaling to unit variance.

    Parameters:
    - x: array-like or pandas.DataFrame, the input features to scale.

    Returns:
    - The scaled features as a numpy.ndarray.

    This function is a utility for preprocessing data before model training or predictions,
    ensuring that features contribute equally to the analysis.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(x)
