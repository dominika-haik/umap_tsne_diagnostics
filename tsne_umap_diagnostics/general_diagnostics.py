import sklearn.metrics as metrics
from .plotting import plot_distances, plot_similarities
from .umap_diagnostics import calculate_V_matrix, calculate_W_matrix
from .tsne_diagnostics import calculate_P_matrix, calculate_Q_matrix

def distance_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None, title='Fit plot of the original and embedded distances', ax=None):
    if X_original is not None and X_embedded is not None:
        distances_original = metrics.pairwise_distances(X_original)
        distances_embedded = metrics.pairwise_distances(X_embedded)

    return plot_distances(distances_original, distances_embedded, title=title, ax=ax)

def matrix_fit_plot_from_matrices(hd_matrix, ld_matrix, asymmetric_matrix=False, title='Fit plot of similarities in high and low dimensions', ax=None):
    return plot_similarities(hd_matrix=hd_matrix, ld_matrix=ld_matrix, asymmetric_matrix=asymmetric_matrix, title=title, ax=ax)

def matrix_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None,
                    method='tsne', asymmetric_matrix=False, umap_knn_indices=None, umap_approx_W=False,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, min_dist=0.1, spread=1.0,
                    title='Fit plot of the similarities in original and embedded space', ax=None):
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