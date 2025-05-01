import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import squareform

def plot_distances(X_original, X_embedded, title, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    X_original = _upper_tri(X_original)
    X_embedded = _upper_tri(X_embedded)

    data = pd.DataFrame({
        'Original Distance': X_original,
        'Reduced Distance': X_embedded
    })

    sns.scatterplot(data=data, x='Original Distance', y='Reduced Distance', alpha=0.2, ax=ax)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("High-dimensional Distance")
    ax.set_ylabel("Low-dimensional Distance")
    return fig if created_fig else None

def plot_similarities(hd_matrix, ld_matrix, asymmetric_matrix, title, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    if not asymmetric_matrix:
        hd_matrix = _upper_tri(hd_matrix)
        ld_matrix = _upper_tri(ld_matrix)

    data = pd.DataFrame({
        'Original Similarity': hd_matrix.flatten(),
        'Similarity in Embedding': ld_matrix.flatten()
    })

    sns.scatterplot(data=data, x='Original Similarity', y='Similarity in Embedding', alpha=0.2, ax=ax)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("High-dimensional Similarity")
    ax.set_ylabel("Low-dimensional Similarity")
    return fig if created_fig else None


def matrix_heatmap(matrix, title='Matrix heatmap', vmin=None, vmax=None, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    #matrix = _hierarchical_sort(matrix)

    sns.heatmap(matrix, cmap='YlGnBu', annot=False, fmt='.2f', cbar_kws={'label': 'Similarity'}, square=True,
                xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    return fig if created_fig else None

def _upper_tri(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def _hierarchical_sort_order(matrix):
    D = 1 - matrix  # turn similarity into dissimilarity (distance)
    np.fill_diagonal(D, 0)
    D_condense = squareform(D)
    clustering = hierarchy.linkage(D_condense)
    leaves_order = hierarchy.leaves_list(clustering)
    return leaves_order

def _apply_sort_order(matrix, order):
    return matrix[np.ix_(order, order)]

def _hsort(matrix):
    order = _hierarchical_sort_order(matrix)
    return _apply_sort_order(matrix, order)