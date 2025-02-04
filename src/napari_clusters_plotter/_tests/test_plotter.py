import numpy as np
import pytest
import pandas as pd


def create_points(n_samples=100, loc=5):
    from napari.layers import Points

    loc = 5
    n_timeframes = 5
    frame = np.arange(n_timeframes).repeat(n_samples // n_timeframes)
    # make some random points with random features
    points = np.random.random((n_samples, 4))
    points2 = np.random.random((n_samples - 1, 4))

    points[:, 0] = frame
    points2[:, 0] = frame[:-1]

    features = pd.DataFrame({
        'frame': frame,
        'feature1': np.random.normal(size=n_samples, loc=loc),
        'feature2': np.random.normal(size=n_samples, loc=loc),
        'feature3': np.random.normal(size=n_samples, loc=loc),
        'feature4': np.random.normal(size=n_samples, loc=loc),
    })

    features2 = pd.DataFrame({
        'frame': frame[:-1],
        'feature2': np.random.normal(size=n_samples - 1, loc=-loc),
        'feature3': np.random.normal(size=n_samples - 1, loc=-loc),
        'feature4': np.random.normal(size=n_samples - 1, loc=-loc),
    })

    layer1 = Points(points, features=features, size=0.1, blending='translucent_no_depth')
    layer2 = Points(points2, features=features2, size=0.1, translate=(0, 0, 2), blending='translucent_no_depth')

    return layer1, layer2


def create_shapes(n_samples=100):
    from napari.layers import Shapes

    # create 100 random anchors
    np.random.seed(0)
    anchors = np.random.random((n_samples, 2))

    # create 100 random widths and heights
    widths = np.random.random(n_samples)
    heights = np.random.random(n_samples)

    # combine into lists of corner coordinates
    corner1 = anchors - np.c_[widths, heights] / 2
    corner2 = anchors + np.c_[widths, heights] / 2
    corner3 = anchors + np.c_[widths, -heights] / 2
    corner4 = anchors + np.c_[-widths, heights] / 2

    # create a list of polygons
    polygons = np.stack([corner1, corner2, corner3, corner4], axis=1)

    layer1 = Shapes(polygons[:49], shape_type='polygon', edge_color='blue')
    layer2 = Shapes(polygons[50:], shape_type='polygon', edge_color='red')
    features1 = pd.DataFrame({
        'feature1': np.random.normal(size=49),
        'feature2': np.random.normal(size=49),
        'feature3': np.random.normal(size=49),
        'feature4': np.random.normal(size=49),
    })

    features2 = pd.DataFrame({
        'feature1': np.random.normal(size=50),
        'feature2': np.random.normal(size=50),
        'feature3': np.random.normal(size=50),
        'feature4': np.random.normal(size=50),
    })

    layer1.features = features1
    layer2.features = features2

    return layer1, layer2


def create_labels(n_samples=100):
    from napari.layers import Labels
    from skimage import data, measure

    binary_image1 = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.1)
    binary_image2 = data.binary_blobs(length=128, n_dim=3, volume_fraction=0.1)

    labels1 = measure.label(binary_image1)
    labels2 = measure.label(binary_image2)

    n_labels1 = len(np.unique(labels1))
    n_labels2 = len(np.unique(labels2))

    features1 = pd.DataFrame({
        'feature1': np.random.normal(size=n_labels1),
        'feature2': np.random.normal(size=n_labels1),
        'feature3': np.random.normal(size=n_labels1),
        'feature4': np.random.normal(size=n_labels1),
    })

    features2 = pd.DataFrame({
        'feature1': np.random.normal(size=n_labels2),
        'feature2': np.random.normal(size=n_labels2),
        'feature3': np.random.normal(size=n_labels2),
        'feature4': np.random.normal(size=n_labels2),
    })

    layer1 = Labels(labels1, features=features1)
    layer2 = Labels(labels2, features=features2)

    return layer1, layer2


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


@pytest.mark.parametrize("create_data", [create_points, create_shapes])
def test_cluster_export(make_napari_viewer, create_data):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    layer1, layer2 = create_data()
    viewer.add_layer(layer1)
    viewer.add_layer(layer2)