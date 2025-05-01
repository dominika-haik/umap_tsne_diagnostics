import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from .plotting import plot_distances, plot_similarities
from .umap_diagnostics import calculate_V_matrix, calculate_W_matrix, show_V_heatmap, show_W_heatmap
from .tsne_diagnostics import calculate_P_matrix, calculate_Q_matrix, show_P_heatmap, show_Q_heatmap

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

    distance_fit_plot(distances_original, distances_embedded, X_original, X_embedded, ax=ax1)
    matrix_fit_plot(distances_original, distances_embedded, X_original, X_embedded,
                    method, asymmetric_matrix, umap_knn_indices, umap_approx_W,
                    perplexity, n_steps, tolerance, k_neighbours, min_dist, spread,
                    ax=ax2)
    if method == 'tsne':
        show_P_heatmap(distances_original, X_original, perplexity, n_steps, tolerance, vmin=vmin, vmax=vmax, ax=ax3)
        show_Q_heatmap(distances_embedded, X_embedded, vmin=vmin, vmax=vmax, ax=ax4)
    elif method == 'umap':
        show_V_heatmap(distances_original, umap_knn_indices, X_original, k_neighbours, n_steps, tolerance, vmin=vmin, vmax=1, ax=ax3)
        show_W_heatmap(distances_embedded, X_embedded, umap_approx_W, min_dist, spread, vmin=vmin, vmax=1, ax=ax4)
    else:
        raise ValueError('Method must be either "tsne" or "umap"')

    #ax3.collections[-1].colorbar.remove()
    plt.suptitle(title)
    plt.tight_layout()
    return fig, axs