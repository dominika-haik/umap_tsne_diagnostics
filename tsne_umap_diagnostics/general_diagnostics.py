import numpy as np
import sklearn.metrics as metrics
from .plotting import plot_distances

def distance_fit_plot(distances_original=None, distances_embedded=None, X_original=None, X_embedded=None, title='Fit plot of the original and embedded distances'):
    if X_original is not None and X_embedded is not None:
        distances_original = metrics.pairwise_distances(X_original)
        distances_embedded = metrics.pairwise_distances(X_embedded)

    return plot_distances(distances_original, distances_embedded, title=title)