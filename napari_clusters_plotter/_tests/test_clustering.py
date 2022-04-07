import numpy as np
from sklearn import datasets


def test_clustering_widget(make_napari_viewer):

    import napari_clusters_plotter as ncp

    viewer = make_napari_viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    for widget in widget_list:
        _widget = widget(viewer)

        if isinstance(_widget, ncp._clustering.ClusteringWidget):
            plot_widget = _widget

    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == n_wdgts + 1


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

    result = kmeans_clustering(
        measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    # a tuple is returned, where the first item (returned[0]) is the name of
    # the clustering method, and the second one (returned[1]) is predictions
    assert len(np.unique(result[1])) == 2
    assert np.array_equal(1 - true_class, result[1])

    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = kmeans_clustering(
        measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    assert np.isnan(result[n_samples // 2])
    assert np.array_equal(result[~np.isnan(result)], 1 - true_class[~np.isnan(result)])


def test_hdbscan_clustering():

    # create an example dataset
    n_samples = 100
    data = datasets.make_moons(n_samples=n_samples, random_state=1, noise=0.05)
    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import hdbscan_clustering

    min_cluster_size = 5
    min_samples = 2  # number of samples that should be included in one cluster

    result = hdbscan_clustering(
        measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    # a tuple is returned, where the first item (returned[0]) is the name of
    # the clustering method, and the second one is predictions (returned[1])
    assert len(np.unique(result[1])) == 2
    assert np.array_equal(true_class, result[1])

    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = hdbscan_clustering(
        measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    assert np.isnan(result[n_samples // 2])
    assert np.array_equal(result[~np.isnan(result)], true_class[~np.isnan(result)])


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

    result = gaussian_mixture_model(measurements, cluster_number=2)

    assert len(np.unique(result[1])) == n_centers
    assert np.array_equal(true_class, (result[1])) or np.array_equal(
        1 - true_class, (result[1])
    )

    # Test bad data
    true_class[n_samples // 2] = -1
    measurements[n_samples // 2, :] = np.NaN

    result = gaussian_mixture_model(measurements, cluster_number=2)

    assert np.isnan(result[n_samples // 2])

    true_result = true_class[~np.isnan(result)].astype(bool)
    result = result[~np.isnan(result)].astype(bool)

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

    assert np.isnan(result[n_samples // 2])

    true_class = true_class[~np.isnan(result)]
    result = result[~np.isnan(result)]
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


if __name__ == "__main__":
    test_gaussian_mixture_model()
