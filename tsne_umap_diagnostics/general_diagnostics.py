import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import warnings

from .plotting import plot_distances, plot_similarities, _hierarchical_sort_order, _apply_sort_order, matrix_heatmap, plot_cost, plot_score
from .umap_diagnostics import calculate_V_matrix, calculate_W_matrix, umap_individual_cost
from .tsne_diagnostics import calculate_P_matrix, calculate_Q_matrix, tsne_individual_cost

def distance_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None, title='Fit plot of the original and embedded distances', ax=None):
    """
    Plots a scatterplot comparing original high-dimensional distances with reduced low-dimensional distances.

    Parameters:
        distances_original (np.ndarray, optional): Pairwise distances in the original high-dimensional space. If None, they are computed from X_original.
        distances_embedded (np.ndarray, optional): Pairwise distances in the reduced low-dimensional space. If None, they are computed from X_embedded.
        X_original (np.ndarray, optional): Original data points. Used to compute distances if distances_original is None.
        X_embedded (np.ndarray, optional): Embedded data points. Used to compute distances if distances_embedded is None.
        title (str, optional): Title of the plot. Default is 'Fit plot of the original and embedded distances'.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    if X_original is not None and X_embedded is not None:
        distances_original = metrics.pairwise_distances(X_original)
        distances_embedded = metrics.pairwise_distances(X_embedded)

    return plot_distances(distances_original=distances_original, distances_embedded=distances_embedded, title=title, ax=ax)

def matrix_fit_plot_from_matrices(hd_matrix, ld_matrix, asymmetric_matrix=False, title='Fit plot of similarities in high and low dimensions', ax=None):
    """
    Plots a scatterplot comparing original high-dimensional similarities with low-dimensional similarities.

    Parameters:
        hd_matrix (np.ndarray): Similarity matrix in the original high-dimensional space.
        ld_matrix (np.ndarray): Similarity matrix in the reduced low-dimensional space.
        asymmetric_matrix (bool, optional): Whether the similarity matrix is asymmetric. Default is False.
        title (str, optional): Title of the plot. Default is 'Fit plot of similarities in high and low dimensions'.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    return plot_similarities(hd_matrix=hd_matrix, ld_matrix=ld_matrix, asymmetric_matrix=asymmetric_matrix, title=title, ax=ax)

def matrix_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None,
                    method='tsne', asymmetric_matrix=False, umap_knn_indices=None, umap_approx_W=False,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, min_dist=0.1, spread=1.0,
                    title='Fit plot of the similarities in original and embedded space', ax=None):
    """
    Plots a scatterplot comparing similarities in the original and embedded spaces.

    Parameters:
        distances_original (np.ndarray, optional): Pairwise distances in the original high-dimensional space. If None, they are computed from X_original.
        distances_embedded (np.ndarray, optional): Pairwise distances in the reduced low-dimensional space. If None, they are computed from X_embedded.
        X_original (np.ndarray, optional): Original data points. Used to compute distances if distances_original is None.
        X_embedded (np.ndarray, optional): Embedded data points. Used to compute distances if distances_embedded is None.
        method (str, optional): Dimensionality reduction method ('tsne' or 'umap'). Default is 'tsne'.
        asymmetric_matrix (bool, optional): Whether the similarity matrix is asymmetric. Default is False.
        n_steps (int, optional): Number of steps for binary search in similarity computation. Default is 100.

        Parameters specific to t-SNE:
            perplexity (float, optional): Perplexity parameter for t-SNE. Default is 30.
            tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.

        Parameters specific to UMAP:
            umap_knn_indices (np.ndarray, optional): Precomputed k-nearest neighbors indices for UMAP. Default is None.
            umap_approx_W (bool, optional): Whether to use an approximation for the W matrix in UMAP. Default is False.
            k_neighbours (int, optional): Number of neighbors for UMAP. Default is 15.
            min_dist (float, optional): Minimum distance for UMAP similarity computation. Default is 0.1.
            spread (float, optional): Spread parameter for UMAP similarity computation. Default is 1.0.

        title (str, optional): Title of the plot. Default is 'Fit plot of the similarities in original and embedded space'.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    if method != 'umap':
        if k_neighbours != 15 or min_dist != 0.1 or spread != 1.0 or umap_knn_indices is not None or umap_approx_W:
            warnings.warn("You passed UMAP-related parameters, but method is not 'umap'. These will be ignored.")

    if X_original is not None and X_embedded is not None:
        distances_original = metrics.pairwise_distances(X_original)
        distances_embedded = metrics.pairwise_distances(X_embedded)

    if method == 'tsne':
        hd_matrix = calculate_P_matrix(distances_original=distances_original, X_original=X_original, perplexity=perplexity, n_steps=n_steps, tolerance=tolerance)
        ld_matrix = calculate_Q_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded)
    elif method == 'umap':
        hd_matrix = calculate_V_matrix(distances_original=distances_original, indices=umap_knn_indices, X_original=X_original, k_neighbours=k_neighbours, n_steps=n_steps, tolerance=tolerance)
        ld_matrix = calculate_W_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded, use_approximation=umap_approx_W, min_dist=min_dist, spread=spread)
    else:
        raise ValueError('Method must be either "tsne" or "umap"')

    return plot_similarities(hd_matrix=hd_matrix, ld_matrix=ld_matrix, asymmetric_matrix=asymmetric_matrix, title=title, ax=ax)

def diagnostic_plots(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None,
                    method='tsne', asymmetric_matrix=False, umap_knn_indices=None, umap_approx_W=False,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, min_dist=0.1, spread=1.0,
                    title='Diagnostic plots for the quality of dimension reduction', vmin=0, vmax=None):
    """
    Generates diagnostic plots to evaluate the quality of dimensionality reduction.

    Parameters:
        distances_original (np.ndarray, optional): Pairwise distances in the original high-dimensional space. If None, they are computed from X_original.
        distances_embedded (np.ndarray, optional): Pairwise distances in the reduced low-dimensional space. If None, they are computed from X_embedded.
        X_original (np.ndarray, optional): Original data points. Used to compute distances if distances_original is None.
        X_embedded (np.ndarray, optional): Embedded data points. Used to compute distances if distances_embedded is None.
        method (str, optional): Dimensionality reduction method ('tsne' or 'umap'). Default is 'tsne'.
        asymmetric_matrix (bool, optional): Whether the high-dimensional similarity matrix is asymmetric. Default is False.
        n_steps (int, optional): Number of steps for binary search in similarity computation. Default is 100.

        Parameters specific to t-SNE:
            perplexity (float, optional): Perplexity parameter for t-SNE. Default is 30.
            tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.

        Parameters specific to UMAP:
            umap_knn_indices (np.ndarray, optional): Precomputed k-nearest neighbors indices for UMAP. Default is None.
            umap_approx_W (bool, optional): Whether to use an approximation for the W matrix in UMAP. Default is False.
            k_neighbours (int, optional): Number of neighbors for UMAP. Default is 15.
            min_dist (float, optional): Minimum distance for UMAP similarity computation. Default is 0.1.
            spread (float, optional): Spread parameter for UMAP similarity computation. Default is 1.0.

        title (str, optional): Title of the diagnostic plots. Default is 'Diagnostic plots for the quality of dimension reduction'.
        vmin (float, optional): Minimum value for heatmap color scale. Default is 0.
        vmax (float, optional): Maximum value for t-SNE's heatmap color scale. If None, vmax=1/(n_samples*6). For UMAP, vmax=1 always. Default is None.

    Returns:
        tuple: A tuple containing the figure object and an array of axes objects.
    """
    if method != 'umap':
        if k_neighbours != 15 or min_dist != 0.1 or spread != 1.0 or umap_knn_indices is not None or umap_approx_W:
            warnings.warn("You passed UMAP-related parameters, but method is not 'umap'. These will be ignored.")

    if X_original is not None:
        n_samples = X_original.shape[0]
    else:
        n_samples = distances_original.shape[0]

    if vmax is None:
        vmax = 1 / (n_samples * 6) # only for tsne

    fig, axs = plt.subplots(2,2, figsize=(12, 10))
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]

    distance_fit_plot(distances_original=distances_original, distances_embedded=distances_embedded, X_original=X_original, X_embedded=X_embedded, ax=ax1)
    matrix_fit_plot(distances_original=distances_original, distances_embedded=distances_embedded, X_original=X_original, X_embedded=X_embedded,
                    method=method, asymmetric_matrix=asymmetric_matrix, umap_knn_indices=umap_knn_indices, umap_approx_W=umap_approx_W,
                    perplexity=perplexity, n_steps=n_steps, tolerance=tolerance, k_neighbours=k_neighbours, min_dist=min_dist, spread=spread,
                    ax=ax2)
    if method == 'tsne':
        P = calculate_P_matrix(distances_original=distances_original, X_original=X_original, perplexity=perplexity, n_steps=n_steps, tolerance=tolerance, asymmetric=asymmetric_matrix)
        order = _hierarchical_sort_order(P)
        P = _apply_sort_order(P, order)
        Q = calculate_Q_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded)
        Q = _apply_sort_order(Q, order)
        matrix_heatmap(matrix=P, title='P matrix heatmap', vmin=vmin, vmax=vmax, ax=ax3)
        matrix_heatmap(matrix=Q, title='Q matrix heatmap',vmin=vmin, vmax=vmax, ax=ax4)
    elif method == 'umap':
        V = calculate_V_matrix(distances_original=distances_original, indices=umap_knn_indices, X_original=X_original, k_neighbours=k_neighbours, n_steps=n_steps, tolerance=tolerance, asymmetric=asymmetric_matrix)
        order = _hierarchical_sort_order(V)
        V = _apply_sort_order(V, order)
        W = calculate_W_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded, use_approximation=umap_approx_W, min_dist=min_dist, spread=spread)
        W = _apply_sort_order(W, order)
        matrix_heatmap(V, title='V matrix heatmap',vmin=vmin, vmax=1, ax=ax3)
        matrix_heatmap(W, title='W matrix heatmap',vmin=vmin, vmax=1, ax=ax4)
    else:
        raise ValueError('Method must be either "tsne" or "umap"')

    #ax3.collections[-1].colorbar.remove()
    plt.suptitle(title)
    plt.tight_layout()
    return fig, axs


def plot_individual_cost(X_original=None, X_embedded=None, method='tsne', umap_knn_indices=None, umap_approx_W=False,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, min_dist=0.1, spread=1.0, title='Individual cost plot', ax=None):
    """
    Plots the individual Kullback-Leibler (KL) divergence cost for each data point after dimensionality reduction,
    using asymmetric high-dimensional similarity matrices (P or V). High values indicate high difference between
    relative positions of points in the original and embedded space.

    Parameters:
        X_original (np.ndarray, optional): Original high-dimensional data.
        X_embedded (np.ndarray, optional): Embedded one- or two-dimensional data.
        method (str, optional): Dimensionality reduction method ('tsne' or 'umap'). Default is 'tsne'.
        title (str, optional): Title of the plot. Default is 'Individual cost plot'.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        n_steps (int, optional): Number of steps for binary search in similarity computation. Default is 100.

        t-SNE specific:
            perplexity (float, optional): Perplexity parameter for t-SNE. Default is 30.
            tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.

        UMAP specific:
            umap_knn_indices (np.ndarray, optional): Precomputed k-nearest neighbors indices for UMAP. Default is None.
            umap_approx_W (bool, optional): Whether to use an approximation for the W matrix in UMAP. Default is False.
            k_neighbours (int, optional): Number of neighbors for UMAP. Default is 15.
            min_dist (float, optional): Minimum distance for UMAP similarity computation. Default is 0.1.
            spread (float, optional): Spread parameter for UMAP similarity computation. Default is 1.0.

    Returns:
        matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
    """
    if method != 'umap':
        if k_neighbours != 15 or min_dist != 0.1 or spread != 1.0 or umap_knn_indices is not None or umap_approx_W:
            warnings.warn("You passed UMAP-related parameters, but method is not 'umap'. These will be ignored.")

    if X_embedded.shape[1] > 2:
        raise ValueError('X_embedded must be one- or two-dimensional')

    if method == 'tsne':
        # Asymmetric hd matrix
        hd_matrix = calculate_P_matrix(distances_original=None, X_original=X_original,
                                       perplexity=perplexity, n_steps=n_steps, tolerance=tolerance, asymmetric=True)
        ld_matrix = calculate_Q_matrix(distances_embedded=None, X_embedded=X_embedded)
        cost = tsne_individual_cost(hd_matrix, ld_matrix)
    elif method == 'umap':
        # Asymmetric hd matrix
        hd_matrix = calculate_V_matrix(distances_original=None, indices=umap_knn_indices,
                                       X_original=X_original, k_neighbours=k_neighbours, n_steps=n_steps,
                                       tolerance=tolerance, asymmetric=True)
        ld_matrix = calculate_W_matrix(distances_embedded=None, X_embedded=X_embedded,
                                       use_approximation=umap_approx_W, min_dist=min_dist, spread=spread)
        # Add small epsilon and normalize to avoid division by zero
        epsilon = 1e-12
        ld_matrix += epsilon
        hd_matrix += epsilon
        ld_matrix /= np.sum(ld_matrix, axis=1, keepdims=True)
        hd_matrix /= np.sum(hd_matrix, axis=1, keepdims=True)
        cost = umap_individual_cost(hd_matrix, ld_matrix)
    else:
        raise ValueError('Method must be either "tsne" or "umap"')

    return plot_cost(X_embedded=X_embedded, cost=cost, title=title, ax=ax)

def outlier_score(hd_matrix):
    """
        Computes an outlier score for each data point by summing the incoming similarities.
        For each column in the asymmetric high-dimensional similarity matrix, the function sums all values,
        resulting in a score that reflects how much other points consider a given point to be their neighbour in the original dataset.
        Lower scores may indicate outliers, while higher scores indicate points that are more similar to others.

        Parameters:
            hd_matrix (np.ndarray): High-dimensional similarity matrix (2D array).

        Returns:
            np.ndarray: 1D array of outlier scores, one for each data point (sum of incoming similarities per point).
        """
    return np.sum(hd_matrix, axis=0)

def plot_outlier_score(X_original=None, X_embedded=None, method='tsne', umap_knn_indices=None,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, title='Outlier score plot', ax=None, vmin=0, vmax=0.3):
    """
        Plots the outlier score for each data point after dimensionality reduction, using the sum of incoming similarities
        in the high-dimensional space. Points with lower scores are more likely to be outliers.

        Parameters:
            General:
                X_original (np.ndarray, optional): Original high-dimensional data.
                X_embedded (np.ndarray, optional): Embedded one- or two-dimensional data.
                method (str, optional): Dimensionality reduction method ('tsne' or 'umap'). Default is 'tsne'.
                n_steps (int, optional): Number of steps for binary search in similarity computation. Default is 100.
                title (str, optional): Title of the plot. Default is 'Outlier score plot'.
                ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
                vmin (float, optional): Minimum value for the color scale. Default is 0.
                vmax (float, optional): Maximum value for the color scale. Default is 0.3, because outlier are expected to score below 0.3.

            t-SNE specific:
                perplexity (float, optional): Perplexity parameter for t-SNE. Default is 30.
                tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.

            UMAP specific:
                umap_knn_indices (np.ndarray, optional): Precomputed k-nearest neighbors indices for UMAP. Default is None.
                k_neighbours (int, optional): Number of neighbors for UMAP. Default is 15.

        Returns:
            matplotlib.figure.Figure or None: The figure object if a new figure is created, otherwise None.
        """
    if method != 'umap':
        if k_neighbours != 15 or umap_knn_indices is not None:
            warnings.warn("You passed UMAP-related parameters, but method is not 'umap'. These will be ignored.")

    if X_embedded.shape[1] > 2:
        raise ValueError('X_embedded must be one- or two-dimensional')

    if method == 'tsne':
        # Asymmetric hd matrix
        hd_matrix = calculate_P_matrix(distances_original=None, X_original=X_original,
                                       perplexity=perplexity, n_steps=n_steps, tolerance=tolerance, asymmetric=True)
    elif method == 'umap':
        # Asymmetric hd matrix
        hd_matrix = calculate_V_matrix(distances_original=None, indices=umap_knn_indices,
                                       X_original=X_original, k_neighbours=k_neighbours, n_steps=n_steps,
                                       tolerance=tolerance, asymmetric=True)
    else:
        raise ValueError('Method must be either "tsne" or "umap"')

    score = outlier_score(hd_matrix)
    return plot_score(X_embedded=X_embedded, score=score, title=title, ax=ax, vmin=vmin, vmax=vmax)
