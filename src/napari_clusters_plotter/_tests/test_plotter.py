import numpy as np
import pandas as pd

import pytest

np.random.seed(0)

def create_random_features(n_samples=100, n_features=5):
    features = pd.DataFrame(
        {f"feature{i}": np.random.random(n_samples) for i in range(n_features)}
    )

    return features

def create_random_points_data(n_samples=100):
    from napari.layers import Points
    points = np.random.random((n_samples, 3)) * 1000
    features = create_random_features(n_samples)
    points_layer = Points(points, features=features)

    return points_layer

def create_random_shapes_data(n_samples=20):
    from napari.layers import Shapes
    box_centers = np.random.random((n_samples, 2)) * 1000
    box_widths = np.random.random(n_samples) * 100
    box_heights = np.random.random(n_samples) * 100
    features = create_random_features(n_samples)

    boxes = []
    for i in range(n_samples):
        corners = np.array(
            [
                [
                    box_centers[i, 0] - box_widths[i] / 2,
                    box_centers[i, 1] - box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] + box_widths[i] / 2,
                    box_centers[i, 1] - box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] + box_widths[i] / 2,
                    box_centers[i, 1] + box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] - box_widths[i] / 2,
                    box_centers[i, 1] + box_heights[i] / 2,
                ],
            ]
        )
        boxes.append(corners)

    shapes_layer = Shapes(boxes, shape_type="polygon", features=features)

    return shapes_layer

def create_random_labels_data():
    from skimage import data, measure
    from napari.layers import Labels

    blobs_3d = data.binary_blobs(length=128, n_dim=3, rng=42)
    labels = measure.label(blobs_3d)
    features = create_random_features(labels.max() + 1)
    labels_layer = Labels(labels, features=features)

    return labels_layer

def create_random_vectors_data(n_samples=10):
    from napari.layers import Vectors

    random_points = np.random.random((n_samples, 3)) * 1000
    random_directions = np.random.uniform(-1, 1, (n_samples, 3))

    random_vectors = np.stack([random_points, random_points + random_directions], axis=1)
    features = create_random_features(n_samples)
    vectors_layer = Vectors(random_vectors, features=features)

    return vectors_layer


@pytest.fixture(params=[
    create_random_points_data,
    create_random_shapes_data,
    create_random_labels_data,
    create_random_vectors_data
])
def random_layer(request):
    return request.param()

def test_adding_layers(make_napari_viewer, random_layer):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    viewer.add_layer(random_layer)
    assert len(viewer.layers) == 1

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget)

    # check that the features are correctly retrieved through the plotter widget
    features = widget._get_features()

    # Make sure that Plotter inserted 'layer' column that indicates which layer the
    # feature was retrieved from
    assert 'layer' in features.columns

    # make sure that the features are the same as the ones in the layer
    for column in random_layer.features.columns:
        assert np.array_equal(features[column], random_layer.features[column])

def test_cluster_selection(make_napari_viewer, random_layer):
    # test that objects are colored correctly when clusters are selected
    from napari_clusters_plotter import PlotterWidget
    from napari.layers import Points, Shapes, Labels, Vectors

    viewer = make_napari_viewer()
    viewer.add_layer(random_layer)
    assert len(viewer.layers) == 1

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget)

    widget._selectors['x'].setCurrentIndex(0)
    widget._selectors['y'].setCurrentIndex(1)
    n_samples = len(widget.plotting_widget.active_artist.color_indices)
    clusters = np.random.randint(0, 3, size=n_samples)
    widget.plotting_widget.active_artist.color_indices = clusters

    colormap = widget.plotting_widget.active_artist.categorical_colormap
    colors = colormap(clusters)

    if isinstance(random_layer, Points):
        assert np.allclose(random_layer.face_color, colors)

    elif isinstance(random_layer, Shapes):
        assert np.allclose(random_layer.face_color, colors)

    elif isinstance(random_layer, Labels):
        pass
        #TODO implement test for Labels

    elif isinstance(random_layer, Vectors):
        assert np.allclose(random_layer.edge_color, colors)

def test_multi_layer_selection(make_napari_viewer):
    # This test adds multiple layer of the same type and checks that the features
    # are correctly retrieved when selecting multiple layers
    pass

def test_cached_cluster_selection(make_napari_viewer):
    # This test adds multiple layers, draw a selection on the first layer,
    # selects a different layer and then re-selects the first layer. The
    # previous selection should still be present.
    pass


if __name__ == "__main__":
    import napari
    test_cluster_selection(napari.Viewer, create_random_points_data())
