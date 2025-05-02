import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from .plotting import plot_distances, plot_similarities, _hierarchical_sort_order, _apply_sort_order, matrix_heatmap
from .umap_diagnostics import calculate_V_matrix, calculate_W_matrix
from .tsne_diagnostics import calculate_P_matrix, calculate_Q_matrix

def distance_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None, title='Fit plot of the original and embedded distances', ax=None):
    if X_original is not None and X_embedded is not None:
        distances_original = metrics.pairwise_distances(X_original)
        distances_embedded = metrics.pairwise_distances(X_embedded)

    return plot_distances(distances_original=distances_original, distances_embedded=distances_embedded, title=title, ax=ax)

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

def diagnostic_plots(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None,
                    method='tsne', asymmetric_matrix=False, umap_knn_indices=None, umap_approx_W=False,
                    perplexity=30, n_steps=100, tolerance = 1e-5, k_neighbours=15, min_dist=0.1, spread=1.0,
                    title='Diagnostic plots for the quality of dimension reduction', vmin=0, vmax=None):

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
        P = calculate_P_matrix(distances_original=distances_original, X_original=X_original, perplexity=perplexity, n_steps=n_steps, tolerance=tolerance)
        order = _hierarchical_sort_order(P)
        P = _apply_sort_order(P, order)
        Q = calculate_Q_matrix(distances_embedded=distances_embedded, X_embedded=X_embedded)
        Q = _apply_sort_order(Q, order)
        matrix_heatmap(matrix=P, title='P matrix heatmap', vmin=vmin, vmax=vmax, ax=ax3)
        matrix_heatmap(matrix=Q, title='Q matrix heatmap',vmin=vmin, vmax=vmax, ax=ax4)
    elif method == 'umap':
        V = calculate_V_matrix(distances_original=distances_original, indices=umap_knn_indices, X_original=X_original, k_neighbours=k_neighbours, n_steps=n_steps, tolerance=tolerance)
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