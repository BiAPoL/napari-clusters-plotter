import numpy as np
import pandas as pd
from napari import Viewer
from sklearn import datasets

from .._utilities import set_features


def test_clustering_widget(make_napari_viewer):
    import napari_clusters_plotter as ncp

    viewer: Viewer = make_napari_viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    for widget in widget_list:
        _widget = widget(viewer)

        if isinstance(_widget, ncp._clustering.ClusteringWidget):
            cluster_widget = _widget

    viewer.window.add_dock_widget(cluster_widget)
    assert len(viewer.window._dock_widgets) == n_wdgts + 1

    label_data = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2, 2],
            [3, 3, 0, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 5, 5],
            [6, 6, 6, 6, 0, 5, 0],
            [0, 7, 7, 0, 0, 0, 0],
        ]
    )

    viewer.add_labels(label_data, name="label_1")
    labels_2 = viewer.add_labels(label_data, name="label_2")

    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(
        n_samples=n_samples,
        random_state=1,
        centers=n_centers,
        cluster_std=0.3,
        n_features=2,
    )

    dataframe = pd.DataFrame(data[0], columns=["x", "y"])
    set_features(labels_2, dataframe)

    cluster_widget.layer_select.value = labels_2
    cluster_widget.clust_method_choice_list.value = "KMeans"
    cluster_widget.run(
        cluster_widget.layer_select.value,
        [i.text() for i in cluster_widget.properties_list.selectedItems()],
        cluster_widget.clust_method_choice_list.current_choice,
        cluster_widget.kmeans_nr_clusters.value,
        cluster_widget.kmeans_nr_iterations.value,
        cluster_widget.standardization.value,
        cluster_widget.hdbscan_min_clusters_size.value,
        cluster_widget.hdbscan_min_nr_samples.value,
        cluster_widget.gmm_nr_clusters.value,
        cluster_widget.ms_quantile.value,
        cluster_widget.ms_n_samples.value,
        cluster_widget.ac_n_clusters.value,
        cluster_widget.ac_n_neighbors.value,
        cluster_widget.custom_name.text(),
        show=False,
    )


def test_kmeans_clustering():
    # create an example dataset
    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(
        n_samples=n_samples,
        random_state=1,
        centers=n_centers,
        cluster_std=0.3,
        n_features=2,
    )

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import kmeans_clustering

    # kmeans_clustering returns (str, np.ndarray), where the first item is algorithm name
    result = kmeans_clustering(
        measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    assert len(np.unique(result[1])) == 2
    assert np.array_equal(1 - true_class, result[1])

    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = kmeans_clustering(
        measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    assert np.isnan(result[1][n_samples // 2])
    assert np.array_equal(
        result[1][~np.isnan(result[1])], 1 - true_class[~np.isnan(result[1])]
    )


def test_hdbscan_clustering():
    # create an example dataset
    n_samples = 100
    data = datasets.make_moons(n_samples=n_samples, random_state=1, noise=0.05)
    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import hdbscan_clustering

    min_cluster_size = 5
    min_samples = 2  # number of samples that should be included in one cluster

    # hdbscan_clustering returns (str, np.ndarray), where the first item is algorithm name
    result = hdbscan_clustering(
        measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    assert len(np.unique(result[1])) == 2
    assert np.array_equal(true_class, result[1])

    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = hdbscan_clustering(
        measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    assert np.isnan(result[1][n_samples // 2])
    assert np.array_equal(
        result[1][~np.isnan(result[1])], true_class[~np.isnan(result[1])]
    )


def test_gaussian_mixture_model():
    # create an example dataset
    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(
        n_samples=n_samples,
        random_state=1,
        centers=n_centers,
        cluster_std=0.3,
        n_features=2,
    )

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import gaussian_mixture_model

    # gaussian_mixture_model returns (str, np.ndarray), where the first item is algorithm name
    result = gaussian_mixture_model(measurements, cluster_number=2)
    print(result[1])

    assert len(np.unique(result[1])) == n_centers
    assert np.array_equal(true_class, (result[1])) or np.array_equal(
        1 - true_class, (result[1])
    )

    # Test bad data
    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = gaussian_mixture_model(measurements, cluster_number=2)

    assert np.isnan(result[1][n_samples // 2])

    true_result = true_class[~np.isnan(result[1])].astype(bool)
    result = result[1][~np.isnan(result[1])].astype(bool)

    assert np.array_equal(result, 1 - true_result) or np.array_equal(
        result, true_result
    )


def test_agglomerative_clustering():
    # create an example dataset
    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(
        n_samples=n_samples,
        random_state=1,
        centers=n_centers,
        cluster_std=0.3,
        n_features=2,
    )

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import agglomerative_clustering

    result = agglomerative_clustering(measurements, cluster_number=2, n_neighbors=2)

    assert len(np.unique(result[1])) == n_centers
    assert np.array_equal(true_class, (result[1])) or np.array_equal(
        1 - true_class, (result[1])
    )

    # Test bad data
    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = agglomerative_clustering(measurements, cluster_number=2, n_neighbors=2)

    assert np.isnan(result[1][n_samples // 2])

    true_class = true_class[~np.isnan(result[1])]
    result = result[1][~np.isnan(result[1])]
    assert np.array_equal(true_class, result) or np.array_equal(1 - true_class, result)


def test_mean_shift():
    # create an example dataset
    n_samples = 20
    n_centers = 2
    data = datasets.make_blobs(
        n_samples=n_samples,
        random_state=1,
        centers=n_centers,
        cluster_std=0.3,
        n_features=2,
    )

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import mean_shift

    result = mean_shift(measurements, quantile=0.5, n_samples=50)

    assert len(np.unique(result[1])) == n_centers
    assert np.array_equal(true_class, result[1]) or np.array_equal(
        1 - true_class, result[1]
    )

    # Test bad data
    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN
    result = mean_shift(measurements, quantile=0.5, n_samples=50)

    assert np.isnan(result[1][n_samples // 2])
    assert np.array_equal(
        result[1][~np.isnan(result[1])], 1 - true_class[~np.isnan(result[1])]
    )
