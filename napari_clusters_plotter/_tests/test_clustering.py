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

def test_clustering_bad_data(make_napari_viewer):
    viewer = make_napari_viewer()
    from napari_clusters_plotter._clustering import ClusteringWidget
    from napari_clusters_plotter._measure import get_regprops_from_regprops_source
    from napari_clusters_plotter._utilities import set_features
    from napari_clusters_plotter._dimensionality_reduction import run_dimensionality_reduction

    label = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2, 2],
            [3, 3, 0, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0],
            [6, 6, 6, 6, 0, 5, 0],  # <-single pixel label
            [0, 7, 7, 0, 0, 0, 0],
        ]
    )

    image = np.random.random((label.shape))
    labels_layer = viewer.add_labels(label)

    widget = ClusteringWidget(viewer)
    measurements = get_regprops_from_regprops_source(image, label, 'shape + intensity')
    set_features(labels_layer, measurements)

    # Reduce dimensionality
    run_dimensionality_reduction(
        labels_layer=labels_layer,
        selected_measurements_list=list(measurements.keys()),
        n_neighbours=15,
        perplexity=5,
        selected_algorithm='UMAP',
        standardize=True,
        explained_variance=5,
        pca_components=2,
        n_components=2)

    widget = ClusteringWidget(viewer)
    widget.run(labels_layer=labels_layer,
            selected_measurements_list = ['UMAP_0', 'UMAP_1'],
            selected_method='HDBSCAN',
            num_clusters=2,
            num_iterations=100,
            standardize=True,
            min_cluster_size=2,
            min_nr_samples=1)

    assert 'HDBSCAN_CLUSTER_ID_SCALER_True' in list(labels_layer.properties.keys())

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

    # test without standardization
    result = kmeans_clustering(
        standardize=False,
        measurements=measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    assert len(np.unique(result)) == n_centers
    assert np.array_equal(1 - true_class, result)

    # test with standardization
    result = kmeans_clustering(
        standardize=True,
        measurements=measurements,
        cluster_number=n_centers,
        iterations=50,
    )

    assert len(np.unique(result)) == n_centers
    assert np.array_equal(1 - true_class, result)


def test_hdbscan_clustering():

    # viewer = napari.Viewer()
    # widget_list = ncp.napari_experimental_provide_dock_widget()
    # n_wdgts = len(viewer.window._dock_widgets)

    # create an example dataset
    n_samples = 100
    data = datasets.make_moons(n_samples=n_samples, random_state=1, noise=0.05)

    true_class = data[1]
    measurements = data[0]

    from napari_clusters_plotter._clustering import hdbscan_clustering

    min_cluster_size = 5
    min_samples = 2  # number of samples that should be included in one cluster

    # test without standardization
    result = hdbscan_clustering(
        standardize=False,
        measurements=measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    assert len(np.unique(result)) == 2
    assert np.array_equal(true_class, result)

    # test with standardization
    result = hdbscan_clustering(
        standardize=True,
        measurements=measurements,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    assert len(np.unique(result)) == 2
    assert np.array_equal(true_class, result)


if __name__ == "__main__":
    import napari
    test_clustering_bad_data(napari.Viewer)
