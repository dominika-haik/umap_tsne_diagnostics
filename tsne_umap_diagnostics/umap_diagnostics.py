import numpy as np
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from .plotting import matrix_heatmap

def calculate_V_matrix(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5):
    if X_original is not None:
        nn = NearestNeighbors(n_neighbors=k_neighbours, metric='euclidean')
        nn.fit(X_original)
        distances_original, indices = nn.kneighbors(X_original)

    n = distances_original.shape[0]
    desired_value = np.log2(k_neighbours)  # for binary search
    distances = distances_original[:, 1:]  # Remove the first column (self-reference) from distances
    indices = indices[:, 1:]      # Remove the first column (self-reference) from indices
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
    V = V + V.T - V * V.T
    return V

def show_V_heatmap(distances_original=None, indices=None, X_original=None ,k_neighbours=15, n_steps=100, tolerance = 1e-5, title='V matrix heatmap'):
    V = calculate_V_matrix(distances_original, indices, X_original ,k_neighbours, n_steps, tolerance)
    return matrix_heatmap(V, title)