import numpy as np
import sklearn.metrics as metrics
from .plotting import matrix_heatmap, _hsort

def calculate_P_matrix(distances_original=None, X_original=None, perplexity=30, n_steps=100, tolerance = 1e-5, asymmetric=False):
    """
    Calculates the P matrix, which represents pairwise similarities in the original space.

    Parameters:
        distances_original (np.ndarray, optional): Precomputed pairwise distances between points. If None, distances are computed from X_original.
        X_original (np.ndarray, optional): Original data points. If provided, distances are computed using Euclidean distance.
        perplexity (float, optional): Desired perplexity value for the similarity computation. Default is 30.
        n_steps (int, optional): Number of steps for binary search to optimize variance. Default is 100.
        tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.
        asymmetric (bool, optional): If True, returns the asymmetric P matrix. Default is False.

    Returns:
        np.ndarray: The symmetric or asymmetric P matrix.
    """
    if X_original is not None:
        distances_original = metrics.pairwise_distances(X_original)

    n_samples = distances_original.shape[0]
    desired_entropy = np.log2(perplexity)
    P = np.zeros((n_samples, n_samples))
    sq_distances = distances_original ** 2

    for i in range(n_samples):
        min_value = -np.inf
        max_value = np.inf
        variance = 1 / np.mean(sq_distances[i, :])

        this_sq_distances = sq_distances[i, :]
        this_sq_distances[i] = np.inf  # exp(-inf) = 0

        # Binary search to optimize variance
        for _ in range(n_steps):
            # Compute conditional probabilities
            nominator = np.exp((-1 * this_sq_distances) / (2 * variance))
            denominator = np.sum(nominator)

            if denominator == 0:
                P[i, :] = 0
            else:
                P[i, :] = nominator / denominator

            # Calculate entropy
            mask = P[i, :] != 0
            entropy = -np.sum(P[i, mask] * np.log2(P[i, mask]))

            entropy_diff = entropy - desired_entropy
            if np.abs(entropy_diff) <= tolerance:
                break

            # Adjust variance and bounds
            if entropy_diff < 0:  # Entropy too small, increase variance
                min_value = variance
                if max_value == np.inf:
                    variance *= 2.0
                else:
                    variance = (variance + max_value) / 2.0
            else:   # Entropy too large, decrease variance
                max_value = variance
                if min_value == -np.inf:
                    variance /= 2.0
                else:
                    variance = (variance + min_value) / 2.0
    if asymmetric:
        return P
    P = (P + P.T) / (2 * n_samples)
    return P

def get_P_heatmap(distances_original=None, X_original=None, perplexity=30, n_steps=100, tolerance = 1e-5, title='P matrix heatmap', vmin=None, vmax=None, ax=None):
    """
    Generates a heatmap of the P matrix and returns the figure object.

    Parameters:
        distances_original (np.ndarray, optional): Precomputed pairwise distances between points. If None, distances are computed from X_original.
        X_original (np.ndarray, optional): Original data points. If provided, distances are computed using Euclidean distance.
        perplexity (float, optional): Desired perplexity value for the similarity computation. Default is 30.
        n_steps (int, optional): Number of steps for binary search to optimize variance. Default is 100.
        tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.
        title (str, optional): Title of the heatmap. Default is 'P matrix heatmap'.
        vmin (float, optional): Minimum value for heatmap color scale. Default is None.
        vmax (float, optional): Maximum value for heatmap color scale. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure object if a new figure is created, otherwise None.
    """
    P = calculate_P_matrix(distances_original=distances_original, X_original=X_original, perplexity=perplexity, n_steps=n_steps, tolerance=tolerance)
    P = _hsort(P)
    return matrix_heatmap(matrix=P, title=title, vmin=vmin, vmax=vmax, ax=ax)

def calculate_Q_matrix(distances_embedded=None, X_embedded=None):
    """
    Calculates the Q matrix, which represents pairwise similarities in the embedded space.

    Parameters:
        distances_embedded (np.ndarray, optional): Precomputed pairwise distances in the embedded space. If None, distances are computed from X_embedded.
        X_embedded (np.ndarray, optional): Embedded data points. If provided, distances are computed using Euclidean distance.

    Returns:
        np.ndarray: The Q matrix, normalized to sum to 1.
    """
    if X_embedded is not None:
        distances_embedded = metrics.pairwise_distances(X_embedded)
    sq_distances_embedded = distances_embedded ** 2

    Q = (1 + sq_distances_embedded) ** (-1)
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q)
    return Q

def get_Q_heatmap(distances_embedded=None, X_embedded=None, title='Q matrix heatmap', vmin=None, vmax=None, ax=None):
    """
    Generates a heatmap of the Q matrix and returns the figure object.

    Parameters:
        distances_embedded (np.ndarray, optional): Precomputed pairwise distances in the embedded space. If None, distances are computed from X_embedded.
        X_embedded (np.ndarray, optional): Embedded data points. If provided, distances are computed using Euclidean distance.
        title (str, optional): Title of the heatmap. Default is 'Q matrix heatmap'.
        vmin (float, optional): Minimum value for heatmap color scale. Default is None.
        vmax (float, optional): Maximum value for heatmap color scale. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure object if a new figure is created, otherwise None.
    """
    Q = calculate_Q_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded)
    Q = _hsort(Q)
    return matrix_heatmap(matrix=Q, title=title, vmin=vmin, vmax=vmax, ax=ax)