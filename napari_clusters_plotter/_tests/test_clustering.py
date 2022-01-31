# -*- coding: utf-8 -*-

import napari
import numpy as np
from sklearn import datasets

import pandas as pd

import sys
sys.path.append('../')


def test_clustering_widget():

    import napari_clusters_plotter as ncp

    viewer = napari.Viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    for widget in widget_list:
        _widget = widget(viewer)

        if isinstance(_widget, ncp._clustering.ClusteringWidget):
            plot_widget = _widget

    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == n_wdgts + 1


def test_kmeans_clustering():

    # viewer = napari.Viewer()
    # widget_list = ncp.napari_experimental_provide_dock_widget()
    # n_wdgts = len(viewer.window._dock_widgets)

    # create an example dataset
    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(n_samples=n_samples, random_state=1,
                               centers=n_centers, cluster_std=0.3, n_features=2)

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import kmeans_clustering

    # test without standardization
    result = kmeans_clustering(standardize=False,
                               measurements=measurements,
                               cluster_number=n_centers,
                               iterations=50)

    assert len(np.unique(result)) == n_centers
    assert np.array_equal(1-true_class, result)

    # test with standardization
    result = kmeans_clustering(standardize=True,
                               measurements=measurements,
                               cluster_number=n_centers,
                               iterations=50)

    assert len(np.unique(result)) == n_centers
    assert np.array_equal(1-true_class, result)

def test_hdbscan_clustering():

    # viewer = napari.Viewer()
    # widget_list = ncp.napari_experimental_provide_dock_widget()
    # n_wdgts = len(viewer.window._dock_widgets)

    # create an example dataset
    n_samples = 100
    data = datasets.make_moons(n_samples=n_samples, random_state=1, noise=0.1)

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import hdbscan_clustering

    min_cluster_size = 5
    min_samples = 60  # number of samples that should be included in one cluster

    # test without standardization
    result = hdbscan_clustering(standardize=False,
                                measurements=measurements,
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples)

    assert len(np.unique(result)) == 2
    assert sum(result == 0) > min_samples

    # test with standardization
    result = hdbscan_clustering(standardize=True,
                                measurements=measurements,
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples)

    assert len(np.unique(result)) == 2
    assert sum(result == 0) > min_samples


if __name__ == "__main__":
    test_kmeans_clustering()