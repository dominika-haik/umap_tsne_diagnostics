import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distances(X_original, X_embedded, title):
    X_original = _upper_tri(X_original)
    X_embedded = _upper_tri(X_embedded)

    data = pd.DataFrame({
        'Original Distance': X_original,
        'Reduced Distance': X_embedded
    })

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Original Distance', y='Reduced Distance', alpha=0.2, ax=ax)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("High-dimensional Distance")
    ax.set_ylabel("Low-dimensional Distance")
    return fig

def plot_similarities(hd_matrix, ld_matrix, asymmetric_matrix, title):
    if not asymmetric_matrix:
        hd_matrix = _upper_tri(hd_matrix)
        ld_matrix = _upper_tri(ld_matrix)

    data = pd.DataFrame({
        'Original Similarity': hd_matrix.flatten(),
        'Similarity in Embedding': ld_matrix.flatten()
    })

    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Original Similarity', y='Similarity in Embedding', alpha=0.2, ax=ax)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("High-dimensional Similarity")
    ax.set_ylabel("Low-dimensional Similarity")
    return fig


def matrix_heatmap(matrix, title='Matrix heatmap'):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, cmap='YlGnBu', annot=False, fmt='.2f', cbar_kws={'label': 'Similarity'},
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    return fig

def _upper_tri(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]
