import numpy as np


def create_multiscale_labels():
    """
    Create a multiscale labels layer with two scales.
    """
    from napari.layers import Labels

    labels = np.array(
        [
            [1, 1, 1, 4, 4, 4],
            [1, 1, 1, 4, 4, 4],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2, 2],
            [3, 3, 0, 2, 2, 2],
            [3, 3, 0, 2, 2, 2],
        ]
    )

    multi_scale_labels = [labels, labels[::2, ::2]]

    layer = Labels(
        multi_scale_labels,
        name="multiscale_labels",
    )

    layer.features["feature1"] = np.random.normal(4)
    layer.features["feature2"] = np.random.normal(4)
    layer.features["feature3"] = np.random.normal(4)
    layer.features["feature4"] = np.random.normal(4)

    return layer


def create_multi_point_layer(n_samples: int = 100):
    import pandas as pd
    from napari.layers import Points

    loc = 5
    n_timeframes = 5
    frame = np.arange(n_timeframes).repeat(n_samples // n_timeframes)
    # make some random points with random  features
    points = np.random.random((n_samples, 4))
    points2 = np.random.random((n_samples - 1, 4))

    points[:, 0] = frame
    points2[:, 0] = frame[:-1]

    features = pd.DataFrame(
        {
            "frame": frame,
            "feature1": np.random.normal(size=n_samples, loc=loc),
            "feature2": np.random.normal(size=n_samples, loc=loc),
            "feature3": np.random.normal(size=n_samples, loc=loc),
            "feature4": np.random.normal(size=n_samples, loc=loc),
        }
    )

    features2 = pd.DataFrame(
        {
            "frame": frame[:-1],
            "feature2": np.random.normal(size=n_samples - 1, loc=-loc),
            "feature3": np.random.normal(size=n_samples - 1, loc=-loc),
            "feature4": np.random.normal(size=n_samples - 1, loc=-loc),
        }
    )

    layer = Points(
        points, features=features, size=0.1, blending="translucent_no_depth"
    )
    layer2 = Points(
        points2,
        features=features2,
        size=0.1,
        translate=(0, 0, 2),
        blending="translucent_no_depth",
    )

    return layer, layer2


def test_mixed_layers(make_napari_viewer):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    random_image = np.random.random((5, 5))
    sample_labels = np.array(
        [
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ]
    )

    viewer.add_image(random_image)
    viewer.add_labels(sample_labels)

    #


def test_cluster_memorization(make_napari_viewer, n_samples: int = 100):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_multi_point_layer(n_samples=n_samples)

    # add layers to viewer
    viewer.add_layer(layer)
    viewer.add_layer(layer2)
    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")

    # select last layer and create a random cluster selection
    viewer.layers.selection.active = layer2
    assert "MANUAL_CLUSTER_ID" in layer2.features.columns

    plotter_widget._selectors["x"].setCurrentText("feature3")
    cluster_indeces = np.random.randint(0, 2, len(layer2.data))
    layer2.features["MANUAL_CLUSTER_ID"] = cluster_indeces
    plotter_widget._selectors["hue"].setCurrentText("MANUAL_CLUSTER_ID")

    # select first layer and make sure that no clusters are selected
    viewer.layers.selection.active = layer
    assert "MANUAL_CLUSTER_ID" in layer.features.columns
    assert np.all(
        plotter_widget.plotting_widget.active_artist.color_indices == 0
    )

    # select last layer and make sure that the clusters are the same
    viewer.layers.selection.active = layer2
    assert np.all(
        plotter_widget.plotting_widget.active_artist.color_indices
        == cluster_indeces
    )


def test_multiscale_plotter(make_napari_viewer):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")

    layer = create_multiscale_labels()
    viewer.add_layer(layer)

    # select some random features in the plotting widget
    plotter_widget._selectors["x"].setCurrentText("feature1")
    plotter_widget._selectors["y"].setCurrentText("feature2")


def test_categorical_handling(make_napari_viewer, n_samples: int = 100):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_multi_point_layer(n_samples=n_samples)

    # add layers to viewer
    viewer.add_layer(layer)
    viewer.add_layer(layer2)
    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")

    # select last layer and create a random cluster selection
    viewer.layers.selection.active = layer2
    assert "MANUAL_CLUSTER_ID" in layer2.features.columns

    categorical_columns = plotter_widget.categorical_columns
    assert (
        len(categorical_columns) == 2
    )  # should only be MANUAL_CLUSTER_ID and layer name
    assert categorical_columns[0] == "MANUAL_CLUSTER_ID"
    assert categorical_columns[1] == "layer"
