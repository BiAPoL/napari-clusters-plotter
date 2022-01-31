# -*- coding: utf-8 -*-

import napari
import numpy as np
from sklearn import datasets
from skimage import measure

import pandas as pd

import sys
sys.path.append('../')
import matplotlib.pyplot as plt


def test_clustering_widget():

    import napari_clusters_plotter as ncp

    viewer = napari.Viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    for widget in widget_list:
        _widget = widget(viewer)

        if isinstance(_widget, ncp._dimensionality_reduction.DimensionalityReductionWidget):
            plot_widget = _widget

    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == n_wdgts + 1


def test_call_to_function():

    viewer = napari.Viewer()

    label = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 2, 2],
                      [0, 0, 0, 0, 2, 2, 2],
                      [3, 3, 0, 0, 0, 0, 0],
                      [0, 0, 4, 4, 0, 5, 5],
                      [6, 6, 6, 6, 0, 5, 0],
                      [0, 7, 7, 0, 0, 0, 0]])

    props = measure.regionprops_table(label, properties=(['label', 'area', 'perimeter']))
    label_layer = viewer.add_labels(label, properties=props)

    from napari_clusters_plotter._dimensionality_reduction import DimensionalityReductionWidget
    from napari_clusters_plotter._utilities import get_layer_tabular_data

    widget = DimensionalityReductionWidget(napari_viewer=viewer)

    widget.run(labels_layer=label_layer,
               selected_measurements_list=['area', 'perimeter'],
               n_neighbours=2,
               perplexity=5,
               selected_algorithm='UMAP',
               standardize=False,
               n_components=2)

    result = get_layer_tabular_data(label_layer)

    assert 'UMAP_0' in result.columns
    assert 'UMAP_1' in result.columns

    widget.run(labels_layer=label_layer,
               selected_measurements_list=['area', 'perimeter'],
               n_neighbours=2,
               perplexity=5,
               selected_algorithm='t-SNE',
               standardize=False,
               n_components=2)

    result = get_layer_tabular_data(label_layer)
    assert 't-SNE_0' in result.columns
    assert 't-SNE_1' in result.columns

def test_umap():

    from napari_clusters_plotter._dimensionality_reduction import umap

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    n_comp = 2

    result = umap(X, n_neigh=2, n_components=n_comp, standardize=True)
    assert result.shape[-1] == n_comp

    result = umap(X, n_neigh=2, n_components=n_comp, standardize=False)
    assert result.shape[-1] == n_comp

def test_tsne():

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    n_comp = 2

    from napari_clusters_plotter._dimensionality_reduction import tsne


    result = tsne(X, perplexity=5, n_components=2, standardize=False)
    assert result.shape[-1] == n_comp

    result = tsne(X, perplexity=5, n_components=2, standardize=True)
    assert result.shape[-1] == n_comp


if __name__ == '__main__':
    test_clustering_widget()
    test_call_to_function()
