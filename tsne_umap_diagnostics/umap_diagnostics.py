import numpy as np
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from .plotting import matrix_heatmap, _hsort

def calculate_V_matrix(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5, asymmetric=False):
    if X_original is not None:
        nn = NearestNeighbors(n_neighbors=k_neighbours, metric='euclidean')
        nn.fit(X_original)
        distances_original, indices = nn.kneighbors(X_original)

    n = distances_original.shape[0]
    desired_value = np.log2(k_neighbours)  # for binary search
    distances = distances_original[:, 1:]  # self-reference
    indices = indices[:, 1:]      # self-reference
    k_neighbours -= 1
    # A 1D array of rho_i
    rhos = distances[:, 0]
    sigmas = np.zeros(n)

    for i in range(n):
        min_value = -np.inf
        max_value = np.inf
        sigmas[i] = 1 / np.mean(distances[i])

        # binary search
        for _ in range(n_steps):
            vi = np.exp((-np.maximum(0, distances[i] - rhos[i])) / sigmas[i])
            entropy = np.sum(vi)

            entropy_diff = entropy - desired_value
            if np.abs(entropy_diff) <= tolerance:
                break

            # Next, adjust variance and bounds (min/max values)
            if entropy_diff < 0:  # entropy too small, increase variance
                min_value = sigmas[i]
                if max_value == np.inf:
                    sigmas[i] *= 2.0
                else:
                    sigmas[i] = (sigmas[i] + max_value) / 2.0
            else:
                max_value = sigmas[i]
                if min_value == -np.inf:
                    sigmas[i] /= 2.0
                else:
                    sigmas[i] = (sigmas[i] + min_value) / 2.0

    similarities = np.exp((-np.maximum(0, distances - rhos[:, np.newaxis])) / sigmas[:, np.newaxis])
    rows = np.repeat(np.arange(n), k_neighbours)
    cols = indices.flatten()
    data = similarities.flatten()
    V = coo_matrix((data, (rows, cols)), shape=(n, n))
    V = V.toarray()
    if asymmetric:
        return V
    V = V + V.T - V * V.T
    return V

def show_V_heatmap(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5, title='V matrix heatmap', vmin=0, vmax=1, ax=None):
    V = calculate_V_matrix(distances_original, indices, X_original ,k_neighbours, n_steps, tolerance)
    V = _hsort(V)
    return matrix_heatmap(V, title, vmin=vmin, vmax=vmax, ax=ax)

def calculate_W_matrix(distances_embedded=None, X_embedded=None, use_approximation=False, min_dist=0.1, spread=1.0):
    if X_embedded is not None:
        distances_embedded = metrics.pairwise_distances(X_embedded, metric='euclidean')

    if use_approximation:
        return approximate_W(distances_embedded=distances_embedded, min_dist=min_dist, spread=spread)

    W = np.where(distances_embedded <= min_dist, 1, np.exp(-distances_embedded - min_dist))
    np.fill_diagonal(W, 0)
    return W

def approximate_W(distances_embedded, min_dist=0.1, spread=1.0):
    distances_sqr = distances_embedded ** 2

    def curve(x, a, b):
        return 1 / (1 + a * x ** (2 * b))

    xdata = np.linspace(0, spread * 3, 300)
    ydata = np.zeros_like(xdata)
    ydata[xdata <= min_dist] = 1
    ydata[xdata > min_dist] = np.exp(-(xdata[xdata > min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xdata, ydata)
    W_approx = curve(distances_sqr, params[0], params[1])
    np.fill_diagonal(W_approx, 0)
    return W_approx

def show_W_heatmap(distances_embedded=None, X_embedded=None, use_approximation=False, min_dist=0.1, spread=1.0, title='W matrix heatmap', vmin=0, vmax=1, ax=None):
    W = calculate_W_matrix(distances_embedded, X_embedded, use_approximation, min_dist, spread)
    W = _hsort(W)
    return matrix_heatmap(W, title, vmin=vmin, vmax=vmax, ax=ax)