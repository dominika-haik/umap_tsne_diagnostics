import numpy as np
import sklearn.metrics as metrics
from .plotting import matrix_heatmap, _hsort

def calculate_P_matrix(distances_original=None, X_original=None, perplexity=30, n_steps=100, tolerance = 1e-5, asymmetric=False):
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

        # binary search
        for _ in range(n_steps):
            # compute conditional probabilities
            nominator = np.exp((-1 * this_sq_distances) / (2 * variance))
            denominator = np.sum(nominator)

            if denominator == 0:
                P[i, :] = 0
            else:
                P[i, :] = nominator / denominator

            # calculate entropy
            mask = P[i, :] != 0
            entropy = -np.sum(P[i, mask] * np.log2(P[i, mask]))

            entropy_diff = entropy - desired_entropy
            if np.abs(entropy_diff) <= tolerance:
                break

            # Next, adjust variance and bounds (min/max values)
            if entropy_diff < 0:  # entropy too small, increase variance
                min_value = variance
                if max_value == np.inf:
                    variance *= 2.0
                else:
                    variance = (variance + max_value) / 2.0
            else:
                max_value = variance
                if min_value == -np.inf:
                    variance /= 2.0
                else:
                    variance = (variance + min_value) / 2.0
    if asymmetric:
        return P
    P = (P + P.T) / (2 * n_samples)
    return P

def show_P_heatmap(distances_original=None, X_original=None, perplexity=30, n_steps=100, tolerance = 1e-5, title='P matrix heatmap', vmin=None, vmax=None, ax=None):
    P = calculate_P_matrix(distances_original, X_original, perplexity, n_steps, tolerance)
    P = _hsort(P)
    return matrix_heatmap(P, title, vmin=vmin, vmax=vmax, ax=ax)

def calculate_Q_matrix(distances_embedded=None, X_embedded=None):
    if X_embedded is not None:
        distances_embedded = metrics.pairwise_distances(X_embedded)
    sq_distances_embedded = distances_embedded ** 2

    Q = (1 + sq_distances_embedded) ** (-1)
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q)
    return Q

def show_Q_heatmap(distances_embedded=None, X_embedded=None, title='Q matrix heatmap', vmin=None, vmax=None, ax=None):
    Q = calculate_Q_matrix(distances_embedded, X_embedded)
    Q = _hsort(Q)
    return matrix_heatmap(Q, title, vmin=vmin, vmax=vmax, ax=ax)