import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import squareform

def plot_distances(distances_original, distances_embedded, title, ax=None):
    """
    Plots a scatterplot comparing original high-dimensional distances with reduced low-dimensional distances.

    Parameters:
        distances_original (np.ndarray): Pairwise distances in the original high-dimensional space.
        distances_embedded (np.ndarray): Pairwise distances in the reduced low-dimensional space.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    distances_original = _upper_tri(distances_original)
    distances_embedded = _upper_tri(distances_embedded)

    data = pd.DataFrame({
        'Original Distance': distances_original,
        'Reduced Distance': distances_embedded
    })

    sns.scatterplot(data=data, x='Original Distance', y='Reduced Distance', alpha=0.2, ax=ax)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("High-dimensional Distance")
    ax.set_ylabel("Low-dimensional Distance")
    return fig if created_fig else None

def plot_similarities(hd_matrix, ld_matrix, asymmetric_matrix, title, ax=None):
    """
    Plots a scatterplot comparing original high-dimensional similarities with low-dimensional similarities.

    Parameters:
        hd_matrix (np.ndarray): Similarity matrix in the original high-dimensional space.
        ld_matrix (np.ndarray): Similarity matrix in the reduced low-dimensional space.
        asymmetric_matrix (bool): Whether the similarity matrix is asymmetric.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
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
    """
    Generates a heatmap for a given matrix.

    Parameters:
        matrix (np.ndarray): The matrix to visualize as a heatmap.
        title (str, optional): Title of the heatmap. Default is 'Matrix heatmap'.
        vmin (float, optional): Minimum value for the heatmap color scale. Default is None.
        vmax (float, optional): Maximum value for the heatmap color scale. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    sns.heatmap(matrix, cmap='YlGnBu', annot=False, fmt='.2f', cbar_kws={'label': 'Similarity'}, square=True,
                xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    return fig if created_fig else None

def _upper_tri(A):
    """
    Extracts the upper triangular part of a square matrix as a flattened array.

    Parameters:
        A (np.ndarray): A square matrix.

    Returns:
        np.ndarray: Flattened array containing the upper triangular part of the matrix.
    """
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def _hierarchical_sort_order(matrix):
    """
    Computes the hierarchical clustering order for a matrix.

    Parameters:
        matrix (np.ndarray): A matrix.

    Returns:
        np.ndarray: The order of leaves after hierarchical clustering.
    """
    D = 1 - matrix  # Convert similarity to dissimilarity (distance)
    np.fill_diagonal(D, 0)
    D_condense = squareform(D)
    clustering = hierarchy.linkage(D_condense, method='complete')
    leaves_order = hierarchy.leaves_list(clustering)
    return leaves_order

def _apply_sort_order(matrix, order):
    """
    Reorders a matrix based on a given order.

    Parameters:
        matrix (np.ndarray): The matrix to reorder.
        order (np.ndarray): The order to apply.

    Returns:
        np.ndarray: The reordered matrix.
    """
    return matrix[np.ix_(order, order)]

def _hsort(matrix):
    """
    Sorts a matrix with hierarchical clustering.

    Parameters:
        matrix (np.ndarray): A matrix.

    Returns:
        np.ndarray: The hierarchically sorted matrix.
    """
    order = _hierarchical_sort_order(matrix)
    return _apply_sort_order(matrix, order)

def plot_cost(X_embedded, cost, title='Cost Plot', ax=None):
    """
        Plots the individual cost values for each data point in the embedded (low-dimensional) space.
        Each point in the embedding is colored according to its associated cost value, allowing for visual identification of points with high or low cost.

        Parameters:
            X_embedded (np.ndarray): Low-dimensional (1D or 2D) embedding of the data (shape: n_samples x n_components).
            cost (array-like): Cost values associated with each data point (length: n_samples).
            title (str, optional): Title of the plot. Default is 'Cost Plot'.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

        Returns:
            matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
        """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    if X_embedded.shape[1] != 2:
        data = pd.DataFrame({
            'x coordinate': X_embedded[:, 0],
            'y coordinate': X_embedded.shape[0] * [1], # Dummy y-coordinate for 1D data
            'cost': cost
        })
    else:
        data = pd.DataFrame({
            'x coordinate': X_embedded[:, 0],
            'y coordinate': X_embedded[:, 1],
            'cost': cost
        })

    # For the colour bar
    norm = mcolors.Normalize(vmin=data['cost'].min(), vmax=data['cost'].max())
    sm = cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])

    sns.scatterplot(data=data, x='x coordinate', y='y coordinate', hue=cost, palette='Reds', hue_norm=norm, alpha=0.7, ax=ax, legend=False)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2" if X_embedded.shape[1] == 2 else "")
    if X_embedded.shape[1] == 1:
        ax.set_yticks([])
    cbar = plt.colorbar(sm, ax=ax)
    ax.set_aspect('equal', adjustable='datalim')

    return fig if created_fig else None

def plot_score(X_embedded, score, title='Outlier Score Plot', ax=None, vmin=0, vmax=0.3):
    """
        Plots the outlier score values for each data point in the embedded (low-dimensional) space.
        Each point in the embedding is colored according to its associated cost value,
        allowing for visual identification of points with high or low likeliness of being outliers.
        Similar to the cost plot, but the colour map is reversed.

        Parameters:
            X_embedded (np.ndarray): Low-dimensional (1D or 2D) embedding of the data (shape: n_samples x n_components).
            score (array-like): Outlier score values associated with each data point (length: n_samples).
            title (str, optional): Title of the plot. Default is 'Outlier Score Plot'.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

        Returns:
            matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
        """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    if X_embedded.shape[1] != 2:
        data = pd.DataFrame({
            'x coordinate': X_embedded[:, 0],
            'y coordinate': X_embedded.shape[0] * [1], # Dummy y-coordinate for 1D data
            'score': score
        })
    else:
        data = pd.DataFrame({
            'x coordinate': X_embedded[:, 0],
            'y coordinate': X_embedded[:, 1],
            'score': score
        })

    # For the colour bar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="Reds_r", norm=norm)
    sm.set_array([])

    sns.scatterplot(data=data, x='x coordinate', y='y coordinate', hue=score, palette='Reds_r', hue_norm=norm, alpha=0.7, ax=ax, legend=False)
    sns.despine()
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2" if X_embedded.shape[1] == 2 else "")
    if X_embedded.shape[1] == 1:
        ax.set_yticks([])
    cbar = plt.colorbar(sm, ax=ax)
    ax.set_aspect('equal', adjustable='datalim')

    return fig if created_fig else None