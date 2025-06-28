import numpy as np
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from .plotting import matrix_heatmap, _hsort

def calculate_V_matrix(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5, asymmetric=False):
    """
    Calculates the V matrix, which represents pairwise similarities between points in the original space.

    Parameters:
        distances_original (np.ndarray, optional): Precomputed pairwise distances between points.
        indices (np.ndarray, optional): Indices of nearest neighbors for each point.
        X_original (np.ndarray, optional): Original data points. If provided, distances and indices are computed using euclidean distance as the metric.
        k_neighbours (int, optional): Number of nearest neighbors to consider. Default is 15.
        n_steps (int, optional): Number of steps for binary search to optimize sigmas. Default is 100.
        tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.
        asymmetric (bool, optional): If True, returns the asymmetric V matrix. Default is False.

    Returns:
        np.ndarray: The symmetric or asymmetric V matrix.
    """
    if X_original is not None:
        nn = NearestNeighbors(n_neighbors=k_neighbours, metric='euclidean')
        nn.fit(X_original)
        distances_original, indices = nn.kneighbors(X_original)

    n = distances_original.shape[0]
    desired_value = np.log2(k_neighbours)  # Target entropy value for binary search
    distances = distances_original[:, 1:]  # Exclude self-reference
    indices = indices[:, 1:]      # Exclude self-reference
    k_neighbours -= 1
    rhos = distances[:, 0]
    sigmas = np.zeros(n)

    for i in range(n):
        min_value = -np.inf
        max_value = np.inf
        sigmas[i] = 1 / np.mean(distances[i])

        # Binary search to optimize sigmas
        for _ in range(n_steps):
            vi = np.exp((-np.maximum(0, distances[i] - rhos[i])) / sigmas[i])
            entropy = np.sum(vi)

            entropy_diff = entropy - desired_value
            if np.abs(entropy_diff) <= tolerance:
                break

            # Adjust variance and bounds
            if entropy_diff < 0:  # Entropy too small, increase variance
                min_value = sigmas[i]
                if max_value == np.inf:
                    sigmas[i] *= 2.0
                else:
                    sigmas[i] = (sigmas[i] + max_value) / 2.0
            else:   # Entropy too large, decrease variance
                max_value = sigmas[i]
                if min_value == -np.inf:
                    sigmas[i] /= 2.0
                else:
                    sigmas[i] = (sigmas[i] + min_value) / 2.0

    # Compute pairwise similarities
    similarities = np.exp((-np.maximum(0, distances - rhos[:, np.newaxis])) / sigmas[:, np.newaxis])
    rows = np.repeat(np.arange(n), k_neighbours)
    cols = indices.flatten()
    data = similarities.flatten()
    V = coo_matrix((data, (rows, cols)), shape=(n, n))
    V = V.toarray()
    if asymmetric:
        return V
    V = V + V.T - V * V.T   # Symmetrize the matrix
    return V

def get_V_heatmap(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5, title='V matrix heatmap', vmin=0, vmax=1, ax=None):
    """
    Generates a heatmap of the V matrix and returns the figure object.

    Parameters:
        distances_original (np.ndarray, optional): Precomputed pairwise distances between points.
        indices (np.ndarray, optional): Indices of nearest neighbors for each point.
        X_original (np.ndarray, optional): Original data points. If provided, distances and indices are computed.
        k_neighbours (int, optional): Number of nearest neighbors to consider. Default is 15.
        n_steps (int, optional): Number of steps for binary search to optimize sigmas. Default is 100.
        tolerance (float, optional): Tolerance for stopping the binary search. Default is 1e-5.
        title (str, optional): Title of the heatmap. Default is 'V matrix heatmap'.
        vmin (float, optional): Minimum value for heatmap color scale. Default is 0.
        vmax (float, optional): Maximum value for heatmap color scale. Default is 1.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure object if a new figure is created, otherwise None.
    """
    V = calculate_V_matrix(distances_original, indices, X_original ,k_neighbours, n_steps, tolerance)
    V = _hsort(V)   # Sort the matrix with hierarchical clustering
    return matrix_heatmap(matrix=V, title=title, vmin=vmin, vmax=vmax, ax=ax)

def calculate_W_matrix(distances_embedded=None, X_embedded=None, use_approximation=False, min_dist=0.1, spread=1.0):
    """
    Calculates the W matrix, which represents pairwise similarities between points in the embedded space.

    Parameters:
        distances_embedded (np.ndarray, optional): Precomputed pairwise distances in the embedded space.
        X_embedded (np.ndarray, optional): Embedded data points. If provided, distances are computed with euclidean distance as the metric.
        use_approximation (bool, optional): If True, uses an approximation for the W matrix. Default is False.
        min_dist (float, optional): Minimum distance for similarity computation. Default is 0.1.
        spread (float, optional): Spread parameter for similarity computation. Default is 1.0.

    Returns:
        np.ndarray: The W matrix.
    """
    if X_embedded is not None:
        distances_embedded = metrics.pairwise_distances(X_embedded, metric='euclidean')

    if use_approximation:
        return approximate_W(distances_embedded=distances_embedded, min_dist=min_dist, spread=spread)

    W = np.where(distances_embedded <= min_dist, 1, np.exp(-(distances_embedded - min_dist)))
    np.fill_diagonal(W, 0)  # Set diagonal to 0 to ignore self-similarity
    return W

def approximate_W(distances_embedded, min_dist=0.1, spread=1.0):
    """
    Approximates the W matrix using a curve-fitting approach.

    Parameters:
        distances_embedded (np.ndarray): Pairwise distances in the embedded space.
        min_dist (float, optional): Minimum distance for similarity computation. Default is 0.1.
        spread (float, optional): Spread parameter for similarity computation. Default is 1.0.

    Returns:
        np.ndarray: The approximated W matrix.
    """
    distances_sqr = distances_embedded ** 2

    def curve(x, a, b):
        return 1 / (1 + a * x ** (2 * b))

    xdata = np.linspace(0, spread * 3, 300)
    ydata = np.zeros_like(xdata)
    ydata[xdata <= min_dist] = 1
    ydata[xdata > min_dist] = np.exp(-(xdata[xdata > min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xdata, ydata)
    W_approx = curve(distances_sqr, params[0], params[1])
    np.fill_diagonal(W_approx, 0)   # Set diagonal to 0 to ignore self-similarity
    return W_approx

def get_W_heatmap(distances_embedded=None, X_embedded=None, use_approximation=False, min_dist=0.1, spread=1.0, title='W matrix heatmap', vmin=0, vmax=1, ax=None):
    """
    Generates a heatmap of the W matrix and returns the figure object.

    Parameters:
        distances_embedded (np.ndarray, optional): Precomputed pairwise distances in the embedded space.
        X_embedded (np.ndarray, optional): Embedded data points. If provided, distances are computed using euclidean distance as the metric.
        use_approximation (bool, optional): If True, uses an approximation for the W matrix. Default is False.
        min_dist (float, optional): Minimum distance for similarity computation. Default is 0.1.
        spread (float, optional): Spread parameter for similarity computation. Default is 1.0.
        title (str, optional): Title of the heatmap. Default is 'W matrix heatmap'.
        vmin (float, optional): Minimum value for heatmap color scale. Default is 0.
        vmax (float, optional): Maximum value for heatmap color scale. Default is 1.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure object if a new figure is created, otherwise None.
    """
    W = calculate_W_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded, use_approximation=use_approximation, min_dist=min_dist, spread=spread)
    W = _hsort(W)   # Sort the matrix with hierarchical clustering
    return matrix_heatmap(matrix=W, title=title, vmin=vmin, vmax=vmax, ax=ax)

def cross_entropy(a, b, j):
    """
    Computes the cross-entropy between two probability distributions, excluding the j-th element.

    Parameters:
        a (np.ndarray): The first probability distribution (e.g., row from V matrix).
        b (np.ndarray): The second probability distribution (e.g., row from W matrix).
        j (int): The index to exclude from the calculation (typically the self-similarity).

    Returns:
        float: The cross-entropy value for the given row, excluding the j-th element.
    """
    mask = np.arange(len(a)) != j
    return np.sum(a[mask] * np.log(a[mask] / b[mask]) + (1 - a[mask]) * np.log((1 - a[mask]) / (1 - b[mask])))

def umap_individual_cost(V, W):
    """
    Computes the individual UMAP cost (cross-entropy) for each data point.

    Parameters:
        V (np.ndarray): The V matrix representing pairwise similarities in the original space.
        W (np.ndarray): The W matrix representing pairwise similarities in the embedded space.

    Returns:
        list: A list of cross-entropy values, one for each data point.
    """
    C = [cross_entropy(V[i], W[i], i) for i in range(len(V))]
    return C