from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model_performance(model_name, y_true, predictions, num_attributes, print_accuracy=False):
    """
    Evaluates the performance of a given model and prints out relevant metrics.
    
    Parameters:
    - model_name: str, the name of the model to evaluate.
    - y_true: array-like, true labels.
    - predictions: array-like, model predictions.
    - num_attriutes: number of attributes needed for adjusted rsquared (X_test.shape[1])
    - print_accuracy: reggresion model - false, classification model (true)
    """

    print(f"Model: {model_name}")
    if(not print_accuracy):
        mae = mean_absolute_error(y_true, predictions)
        print(f"Mean Absolute Error (MAE): {mae}")
        r2 = r2_score(y_true, predictions)
        ar2 = get_rsquared_adj(r2, len(predictions), num_attributes)
        print(f"Adjusted R-squared: {r2}")
    else:
        accuracy = accuracy_score(y_true, predictions)
        print(f'Accuracy: {accuracy:.4f}')

def plot_predictions_distribution(model_name, y_true, predictions):
    """
    Plots the distribution of predictions for a given model.
    
    Parameters:
    - model_name: str, the name of the model for the title of the plot.
    - y_true: array-like, true labels.
    - predictions: array-like, model predictions.
    """
    plt.hist(predictions, bins=len(set(y_true)), edgecolor='black')
    plt.xlabel('League rank')
    plt.ylabel('Number of predictions')
    plt.title(f'Distribution {model_name}')
    plt.show()


def get_rsquared_adj(r_squared, n, p):
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return adjusted_r_squared

