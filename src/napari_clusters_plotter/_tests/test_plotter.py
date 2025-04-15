import numpy as np
import pytest


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


def create_multi_vectors_layer(n_samples: int = 100):
    from napari.layers import Vectors

    points1, points2 = create_multi_point_layer(n_samples=n_samples)

    points_direction1 = np.random.normal(size=points1.data.shape)
    points_direction2 = np.random.normal(size=points2.data.shape)

    # set time index correctly
    points_direction1[:, 0] = points1.data[:, 0]
    points_direction2[:, 0] = points2.data[:, 1]

    vectors1 = np.stack([points1.data, points_direction1], axis=1)
    vectors2 = np.stack([points2.data, points_direction2], axis=1)

    vectors1 = Vectors(vectors1, features=points1.features, name="vectors1")
    vectors2 = Vectors(vectors2, features=points2.features, name="vectors2")

    return vectors1, vectors2


def create_multi_labels_layer():
    from skimage import data, measure
    from napari.layers import Labels
    import pandas as pd

    labels1 = measure.label(data.binary_blobs(length=64, n_dim=2))
    labels2 = measure.label(data.binary_blobs(length=64, n_dim=2))

    features1 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels1.max() + 1),
            "feature2": np.random.normal(size=labels1.max() + 1),
            "feature3": np.random.normal(size=labels1.max() + 1),
        }
    )

    features2 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels2.max() + 1),
            "feature2": np.random.normal(size=labels2.max() + 1),
            "feature3": np.random.normal(size=labels2.max() + 1),
        }
    )

    labels1 = Labels(labels1, name="labels1", features=features1)
    labels2 = Labels(labels2, name="labels2", features=features2, translate=(0, 128))

    return labels1, labels2


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

@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_vectors_layer,
    ],
)
def test_cluster_memorization(make_napari_viewer, create_sample_layers):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_sample_layers()

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


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_vectors_layer,
    ],
)
def test_categorical_handling(make_napari_viewer, create_sample_layers):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_sample_layers()

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
