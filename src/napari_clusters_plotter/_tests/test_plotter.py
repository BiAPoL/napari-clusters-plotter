import numpy as np
import pandas as pd
import pytest
from napari.layers import Labels, Points, Shapes


def create_points(n_samples=100, loc=5):

    loc = 5
    n_timeframes = 5
    frame = np.arange(n_timeframes).repeat(n_samples // n_timeframes)
    # make some random points with random features
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

    layer1 = Points(
        points, features=features, size=0.1, blending="translucent_no_depth"
    )
    layer2 = Points(
        points2,
        features=features2,
        size=0.1,
        translate=(0, 0, 2),
        blending="translucent_no_depth",
    )

    return layer1, layer2


def create_shapes(n_samples=100):

    # create 100 random anchors
    np.random.seed(0)
    anchors = np.random.random((n_samples, 2)) * 100

    # create 100 random widths and heights
    widths = np.random.random(n_samples) * 10
    heights = np.random.random(n_samples) * 10

    # combine into lists of corner coordinates
    corner1 = anchors - np.c_[widths, heights] / 2
    corner2 = anchors + np.c_[widths, -heights] / 2
    corner3 = anchors + np.c_[widths, heights] / 2
    corner4 = anchors + np.c_[-widths, heights] / 2

    # create a list of polygons
    polygons = np.stack([corner1, corner2, corner3, corner4], axis=1)

    layer1 = Shapes(polygons[:49], shape_type="polygon")
    layer2 = Shapes(polygons[50:], shape_type="polygon")
    features1 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=49),
            "feature2": np.random.normal(size=49),
            "feature3": np.random.normal(size=49),
            "feature4": np.random.normal(size=49),
        }
    )

    features2 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=50),
            "feature2": np.random.normal(size=50),
            "feature3": np.random.normal(size=50),
            "feature4": np.random.normal(size=50),
        }
    )

    layer1.features = features1
    layer2.features = features2

    return layer1, layer2


def create_labels(n_samples=100):
    from skimage import data, measure

    binary_image1 = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.1)
    binary_image2 = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.1)

    labels1 = measure.label(binary_image1)
    labels2 = measure.label(binary_image2)

    n_labels1 = len(np.unique(labels1))
    n_labels2 = len(np.unique(labels2))

    features1 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=n_labels1),
            "feature2": np.random.normal(size=n_labels1),
            "feature3": np.random.normal(size=n_labels1),
            "feature4": np.random.normal(size=n_labels1),
        }
    )

    features2 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=n_labels2),
            "feature2": np.random.normal(size=n_labels2),
            "feature3": np.random.normal(size=n_labels2),
            "feature4": np.random.normal(size=n_labels2),
        }
    )

    layer1 = Labels(labels1, features=features1)
    layer2 = Labels(labels2, features=features2)

    return layer1, layer2


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


def create_multi_tracks_layer(n_samples: int = 100):
    from napari.layers import Tracks

    points1, points2 = create_multi_point_layer(n_samples=n_samples)

    tracks1 = points1.data.copy()
    tracks2 = points2.data.copy()

    # insert empty track id column
    tracks1 = np.insert(tracks1, 0, 0, axis=1)
    tracks2 = np.insert(tracks2, 0, 0, axis=1)

    for t in range(int(points1.data[:, 0].max() + 1)):
        # set the track id for each point
        tracks1[tracks1[:, 1] == t, 0] = np.arange(
            len(tracks1[tracks1[:, 1] == t]), dtype=int
        )

    for t in range(int(points2.data[:, 0].max() + 1)):
        # set the track id for each point
        tracks2[tracks2[:, 1] == t, 0] = np.arange(
            len(tracks2[tracks2[:, 1] == t]), dtype=int
        )

    tracks1 = Tracks(tracks1, features=points1.features, name="tracks1")
    tracks2 = Tracks(
        tracks2, features=points2.features, name="tracks2", translate=(0, 0, 2)
    )

    return tracks1, tracks2


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


def create_multi_surface_layer(n_samples: int = 100):
    from napari.layers import Surface

    vertices1, vertices2 = create_multi_point_layer(n_samples=n_samples)

    faces1 = []
    faces2 = []
    for t in range(int(vertices1.data[:, 0].max())):
        vertex_indeces_t = np.argwhere(vertices1.data[:, 0] == t).flatten()

        # draw some random triangles from the indeces
        _faces = np.random.randint(
            low=vertex_indeces_t.min(),
            high=vertex_indeces_t.max(),
            size=(10, 3),
        )
        faces1.append(_faces)

        vertex_indeces_t = np.argwhere(vertices2.data[:, 0] == t).flatten()

        # draw some random triangles from the indeces
        _faces = np.random.randint(
            low=vertex_indeces_t.min(),
            high=vertex_indeces_t.max(),
            size=(10, 3),
        )
        faces2.append(_faces)

    faces1 = np.concatenate(faces1, axis=0)
    faces2 = np.concatenate(faces2, axis=0)

    surface1 = Surface(
        (vertices1.data, faces1),
        features=vertices1.features,
        name="surface1",
    )

    surface2 = Surface(
        (vertices2.data, faces2),
        features=vertices2.features,
        name="surface2",
        translate=(0, 0, 2),
    )
    return surface1, surface2


def create_multi_surface_layer2():
    from napari.layers import Layer

    from napari_clusters_plotter._sample_data import cells3d_curvatures

    layer1 = cells3d_curvatures()[1]
    layer2 = cells3d_curvatures()[1]

    layer1 = Layer.create(*layer1)
    layer2 = Layer.create(*layer2)

    # rename features to feature1, feature2, etc
    layer1.features.columns = [
        f"feature{i}" for i in range(1, len(layer1.features.columns) + 1)
    ]
    layer2.features.columns = [
        f"feature{i}" for i in range(1, len(layer2.features.columns) + 1)
    ]

    return layer1, layer2


def create_multi_shapes_layers(n_samples: int = 100):
    from napari.layers import Shapes

    points1, points2 = create_multi_point_layer(n_samples=n_samples)

    shapes1, shapes2 = [], []
    for i in range(len(points1.data)):
        # create a random shape around the point, whereas the shape
        # consists of the coordinates of the four corner of the rectangle
        y, x = points1.data[i, 2], points1.data[i, 3]
        w, h = np.random.randint(1, 5), np.random.randint(1, 5)

        shape1 = np.array(
            [
                [y - h, x - w],
                [y - h, x + w],
                [y + h, x + w],
                [y + h, x - w],
            ]
        )
        shapes1.append(shape1)

    for i in range(len(points2.data)):
        # create a random shape around the point, whereas the shape consists
        # of the coordinates of the four corner of the rectangle
        y, x = points2.data[i, 2], points2.data[i, 3]
        w, h = np.random.randint(1, 5), np.random.randint(1, 5)

        shape2 = np.array(
            [
                [y - h, x - w],
                [y - h, x + w],
                [y + h, x + w],
                [y + h, x - w],
            ]
        )
        shapes2.append(shape2)

    shape1 = Shapes(shapes1, features=points1.features, name="shapes1")
    shape2 = Shapes(
        shapes2, features=points2.features, name="shapes2", translate=(0, 2)
    )

    return shape1, shape2


def create_multi_labels_layer():
    import pandas as pd
    from napari.layers import Labels
    from skimage import data, measure

    labels1 = measure.label(data.binary_blobs(length=64, n_dim=2))
    labels2 = measure.label(data.binary_blobs(length=64, n_dim=2))

    features1 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels1.max()),
            "feature2": np.random.normal(size=labels1.max()),
            "feature3": np.random.normal(size=labels1.max()),
        }
    )

    features2 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels2.max()),
            "feature2": np.random.normal(size=labels2.max()),
            "feature3": np.random.normal(size=labels2.max()),
        }
    )

    labels1 = Labels(labels1, name="labels1", features=features1)
    labels2 = Labels(
        labels2, name="labels2", features=features2, translate=(0, 128)
    )

    return labels1, labels2


def create_multi_dask_labels_layers():
    import dask.array as da
    import pandas as pd
    from napari.layers import Labels
    from skimage import data, measure

    labels1 = measure.label(data.binary_blobs(length=64, n_dim=2))
    labels2 = measure.label(data.binary_blobs(length=64, n_dim=2))

    # convert labels to dask arrays
    labels1 = da.from_array(labels1, chunks=(32, 32))
    labels2 = da.from_array(labels2, chunks=(32, 32))

    features1 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels1.max()),
            "feature2": np.random.normal(size=labels1.max()),
            "feature3": np.random.normal(size=labels1.max()),
        }
    )

    features2 = pd.DataFrame(
        {
            "feature1": np.random.normal(size=labels2.max()),
            "feature2": np.random.normal(size=labels2.max()),
            "feature3": np.random.normal(size=labels2.max()),
        }
    )

    labels1 = Labels(labels1, name="labels1", features=features1, scale=(1, 1))
    labels2 = Labels(labels2, name="labels2", features=features2, scale=(1, 1))
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


@pytest.mark.parametrize(
    "create_data",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_shapes_layers,
        create_multi_tracks_layer,
    ],
)
def test_cluster_export(make_napari_viewer, create_data):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer1, layer2 = create_data()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)
    viewer.layers.select_all()

    n_layers = len(viewer.layers)

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    for layer in viewer.layers:
        if type(layer) in widget.input_layer_types:
            features = layer.features
            features["MANUAL_CLUSTER_ID"] = np.random.randint(
                low=0, high=2, size=len(features["MANUAL_CLUSTER_ID"])
            )
            layer.features = features

    widget.plot_needs_update.emit()
    widget._on_export_clusters()

    assert len(viewer.layers) == n_layers * 2


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_dask_labels_layers,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_shapes_layers,
        create_multi_tracks_layer,
    ],
)
def test_cluster_memorization_and_clearing(
    make_napari_viewer, create_sample_layers
):
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
    cluster_indeces = np.random.randint(0, 2, len(layer2.features))
    plotter_widget._on_finish_draw(cluster_indeces)

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

    # check that clearing the plot actually sets all "Manual Cluster IDs" to 0
    plotter_widget._reset()
    assert np.all(layer2.features["MANUAL_CLUSTER_ID"] == 0)


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


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_dask_labels_layers,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_shapes_layers,
        create_multi_tracks_layer,
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


def test_empty_layer_clean_up(make_napari_viewer, n_samples: int = 100):
    """
    This test checks what happenns when you add some layers,
    do a manual clustering , then delete the layers and add some others
    """
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()

    points1, points2 = create_multi_point_layer(n_samples=n_samples)
    vectors1, _ = create_multi_vectors_layer(n_samples=n_samples)

    # add points to viewer
    viewer.add_layer(points1)
    viewer.add_layer(points2)

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    viewer.layers.selection.active = points1

    # do a random drawing
    assert "MANUAL_CLUSTER_ID" in points1.features.columns
    random_cluster_indeces = np.random.randint(0, 2, len(points1.data))
    widget._on_finish_draw(random_cluster_indeces)

    # delete the layers
    viewer.layers.clear()

    # check that all widget._selectros ('x', 'y', 'hue') are empty
    assert widget._selectors["x"].currentText() == ""
    assert widget._selectors["y"].currentText() == ""
    assert widget._selectors["hue"].currentText() == ""

    widget._reset()

    # add vectors to viewer
    viewer.add_layer(vectors1)


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_shapes_layers,
    ],
)
def test_temporal_highlighting(make_napari_viewer, create_sample_layers):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_sample_layers()

    # add layers to viewer
    viewer.add_layer(layer)
    viewer.add_layer(layer2)
    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")

    plotter_widget._selectors["x"].setCurrentText("feature3")

    # move time slider
    current_step = viewer.dims.current_step[0]
    viewer.dims.set_current_step(0, current_step + 1)

    # check that the dots in the plotter widget update alpha and size
    # to highlight out-of and in-frame data points
    assert plotter_widget.plotting_widget.active_artist.alpha.min() == 0.25
    assert plotter_widget.plotting_widget.active_artist.size.min() == 35


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_shapes_layers,
    ],
)
def test_histogram_support(make_napari_viewer, create_sample_layers):

    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    layer, layer2 = create_sample_layers()

    # add layers to viewer
    viewer.add_layer(layer)
    viewer.add_layer(layer2)
    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")

    plotter_widget._selectors["x"].setCurrentText("feature3")
    plotter_widget.plotting_type = "HISTOGRAM2D"

    # select both layers
    viewer.layers.selection.active = layer
    assert "MANUAL_CLUSTER_ID" in layer.features.columns
    assert "MANUAL_CLUSTER_ID" in layer2.features.columns

    # trigger manual bin size
    plotter_widget.automatic_bins = False
    plotter_widget.bin_number = 10
    plotter_widget.control_widget.histogram_cmap_box.setCurrentText("viridis")
    plotter_widget.control_widget.overlay_cmap_box.setCurrentText("viridis")

    plotter_widget.plotting_type = "SCATTER"


@pytest.mark.parametrize(
    "create_sample_layers",
    [
        create_multi_point_layer,
        create_multi_labels_layer,
        create_multi_dask_labels_layers,
        create_multi_vectors_layer,
        create_multi_surface_layer,
        create_multi_surface_layer2,
        create_multi_shapes_layers,
    ],
)
def test_cluster_visibility_toggle(make_napari_viewer, create_sample_layers):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3

    # add layers to viewer
    layer, layer2 = create_sample_layers()
    viewer.add_layer(layer)
    viewer.add_layer(layer2)

    plotter_widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")
    plotter_widget._selectors["x"].setCurrentText("feature3")

    plotter_widget._on_show_plot_overlay(state=False)
    plotter_widget._on_show_plot_overlay(state=True)
    plotter_widget._on_show_plot_overlay(state=False)
