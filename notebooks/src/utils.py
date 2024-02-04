import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sb
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def plot_correlation_for_col(df, col_name):
    plt.figure(figsize=(12,6)) # podesimo velicinu grafika
    correlation_matrix = df.corr() # racunamo matricu korelacije
    sorted_col_corr = correlation_matrix[col_name].sort_values(ascending=True) # indeksiramo kolonu i soritramo vrednosti
    sorted_col_corr = sorted_col_corr.drop(col_name) # izbacujemo vrednost samu sa sobom
    sb.barplot(x=sorted_col_corr.index, y=sorted_col_corr.values, palette='RdBu')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_explained_variance(pca_model):
    '''Plots the explained variance plot using a trained PCA model.'''
    plt.figure(figsize=(9,3)) # podesimo velicinu grafika
    
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, align='center')
    plt.xlabel('Glavna komponenta')
    plt.ylabel('Objasnjena varijansa')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, '--o')
    plt.xlabel('Broj glavnih komponenti')
    plt.ylabel('Kumulativna varijansa')

    plt.tight_layout()
    plt.show()

def plot_pc_loading(pca_model, pc_idx, columns, largest_n_pc=None):
    plt.figure(figsize=(12,6)) # podesimo velicinu grafika

    pc_loadings_df = pd.DataFrame(data=pca_model.components_, columns=columns)
    loading = pc_loadings_df.iloc[pc_idx]
    sorted_loading_abs = loading.abs().sort_values(ascending=True)

    largest_n_pc = 0 if largest_n_pc is None else largest_n_pc
    sorted_loading = loading[sorted_loading_abs.index][-largest_n_pc:]
    sb.barplot(x=sorted_loading.index, y=sorted_loading.values, palette='Reds')
    plt.xticks(rotation=90)
    plt.title(f'Correlation with {pc_idx}. component')
    plt.tight_layout()
    plt.show()


def visualize_principal_components(principal_components: np.ndarray, n_principal_components: int, 
                                   target_col: pd.Series = None, n_samples: int = None):
    '''
    Visualizes principal components in 2D or 3D.

    Parameters:
    - principal_components (np.ndarray): The principal components to visualize.
    - n_principal_components (int): Number of principal components (2 or 3).
    - target_col (pd.Series): Target column for color differentiation in the plot.
    - n_samples (int): Number of samples to visualize. If None, all samples are visualized.
    '''

    if n_samples is not None and n_samples < principal_components.shape[0]:
        indices = np.random.choice(principal_components.shape[0], n_samples, replace=False)
        principal_components = principal_components[indices, :]
        if target_col is not None:
            target_col = target_col.iloc[indices]

    if n_principal_components == 2:
        fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1],
                         opacity=0.6, color=target_col, color_continuous_scale='RdBu', width=700, height=600)
        fig.update_traces(marker={'size': 10})
        fig.update_layout(title='Principal components visualisations', xaxis_title="PC1", yaxis_title="PC2")

        fig.show()

    elif n_principal_components == 3:
        fig = px.scatter_3d(x=principal_components[:, 0], y=principal_components[:, 1], z=principal_components[:, 2],
                            opacity=0.6, color=target_col, color_continuous_scale='RdBu', width=1000)
        fig.update_traces(marker={'size': 6})
        fig.update_layout(title='Principal components visualisations', 
                          scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                                     xaxis_autorange='reversed', yaxis_autorange='reversed')) 
        fig.show()

    else:
        raise Exception('number of principal components must be 2 or 3')


