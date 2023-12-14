import sys

import numpy as np
import pandas as pd
from napari.layers import Labels
from qtpy.QtWidgets import QListWidget

import napari_clusters_plotter._utilities as utilities

sys.path.append("../")


class FakeWidget:
    def __init__(self, layer):
        class Layer_select:
            def __init__(self, layer):
                self.value = layer

        self.properties_list = QListWidget()
        self.layer_select = Layer_select(layer)


def test_cluster_image_generation():
    label = np.array(
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

    label_ids = np.array([1, 2, 3, 4, 5, 6, 7])
    predictions = np.array([0, 0, 0, 1, 1, 1, 2])
    result = utilities.generate_cluster_image(label, label_ids, predictions)
    true_result = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 2, 2],
            [2, 2, 2, 2, 0, 2, 0],
            [0, 3, 3, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(result, true_result)

    label_timelapse_3d = np.array([label, label])
    label_timelapse = utilities.reshape_2D_timelapse(label_timelapse_3d)

    label_id_list = np.array([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    predictions_list = np.array([[0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1, 0]])
    result_dask = utilities.dask_cluster_image_timelapse(
        label_timelapse, label_id_list, predictions_list
    )
    true_result_tp2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 2, 2],
            [2, 2, 2, 2, 0, 2, 0],
            [0, 1, 1, 0, 0, 0, 0],
        ]
    )

    assert np.array_equal(result_dask[0].compute()[0], true_result)
    assert np.array_equal(result_dask[1].compute()[0], true_result_tp2)


def test_cluster_image_generation_unsorted_non_sequential_labels():
    label = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2, 2],
            [8, 8, 0, 0, 0, 0, 0],
            [0, 0, 9, 9, 0, 5, 5],
            [6, 6, 6, 6, 0, 5, 0],
            [0, 7, 7, 0, 0, 0, 0],
        ]
    )

    label_ids = np.array([1, 2, 9, 8, 5, 6, 7])
    predictions = np.array([0, 0, 0, 1, 1, 1, 2])
    result = utilities.generate_cluster_image(label, label_ids, predictions)
    true_result = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [2, 2, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 2, 2],
            [2, 2, 2, 2, 0, 2, 0],
            [0, 3, 3, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(result, true_result)

    label_timelapse_3d = np.array([label, label])
    label_timelapse = utilities.reshape_2D_timelapse(label_timelapse_3d)

    label_id_list = np.array([[1, 2, 9, 8, 5, 6, 7], [1, 2, 9, 8, 5, 6, 7]])
    predictions_list = np.array([[0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1, 0]])
    result_dask = utilities.dask_cluster_image_timelapse(
        label_timelapse, label_id_list, predictions_list
    )
    true_result_tp2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [2, 2, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 2, 2],
            [2, 2, 2, 2, 0, 2, 0],
            [0, 1, 1, 0, 0, 0, 0],
        ]
    )

    assert np.array_equal(result_dask[0].compute()[0], true_result)
    assert np.array_equal(result_dask[1].compute()[0], true_result_tp2)


def test_feature_setting(make_napari_viewer):
    viewer = make_napari_viewer()
    label = np.array(
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
    label_layer = viewer.add_labels(label)

    some_features = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    utilities.set_features(label_layer, some_features)

    if hasattr(label_layer, "features"):
        assert isinstance(label_layer.features, pd.DataFrame)
    elif hasattr(label_layer, "properties"):
        assert isinstance(label_layer.properties, dict)

    some_features = utilities.get_layer_tabular_data(label_layer)
    assert isinstance(some_features, pd.DataFrame)

    utilities.add_column_to_layer_tabular_data(label_layer, "C", [5, 6, 7])
    some_features = utilities.get_layer_tabular_data(label_layer)
    assert "C" in some_features.columns

    widget = FakeWidget(label_layer)
    utilities.update_properties_list(widget, exclude_list=["A"])


def test_generate_cluster_image_3D():
    input_labels = [1, 2, 3, 4, 5]
    output_labels = [1, 1, 0, 0, 0]

    label_image_3d = np.array(
        [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]]
    )

    actual_cluster_label_image_3d = utilities.generate_cluster_image(
        label_image_3d, input_labels, output_labels
    )
    expected_cluster_label_image_3d = np.array(
        [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]]
    )

    assert np.array_equal(
        actual_cluster_label_image_3d, expected_cluster_label_image_3d
    )


def test_generate_cluster_tracks():
    label_image_4d_tracking = np.array(
        [
            [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]],
            [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]],
        ]
    )

    plot_cluster_name = "MANUAL_CLUSTER_ID"

    import pandas as pd

    # Creating a DataFrame from a dictionary
    features = {
        "label": [1, 2, 3, 4, 5],
        "MANUAL_CLUSTER_ID": [1, 1, 0, 0, 0],
        "value": [2.0, 3.0, 42.0, 50.0, 60.0],
    }

    features = pd.DataFrame(features)
    labels_layer = Labels(data=label_image_4d_tracking, features=features)
    actual_cluster_layer = np.array(
        utilities.generate_cluster_tracks(labels_layer, plot_cluster_name)
    )
    expected_cluster_layer = np.array(
        [
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
        ]
    )

    assert np.array_equal(actual_cluster_layer, expected_cluster_layer)


def test_generate_cluster_4d_labels():
    label_image_4D_non_tracking = np.array(
        [
            [[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 2, 0], [2, 2, 2], [0, 2, 0]]],
            [[[0, 3, 0], [3, 3, 3], [0, 3, 0]], [[0, 4, 0], [4, 4, 4], [0, 4, 0]]],
        ]
    )

    expected_output = np.array(
        [
            [[[0, 2, 0], [2, 2, 2], [0, 2, 0]], [[0, 2, 0], [2, 2, 2], [0, 2, 0]]],
            [[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
        ]
    )

    plot_cluster_name = "MANUAL_CLUSTER_ID"

    import pandas as pd

    # Creating a DataFrame from a dictionary
    features = {
        "label": [
            1,
            2,
            3,
            4,
        ],
        "MANUAL_CLUSTER_ID": [1, 1, 0, 0],
        "value": [2.0, 3.0, 42.0, 50.0],
        utilities._POINTER: [0, 0, 1, 1],
    }
    features = pd.DataFrame(features)
    labels_layer = Labels(data=label_image_4D_non_tracking, features=features)
    actual_cluster_layer = np.array(
        utilities.generate_cluster_4d_labels(labels_layer, plot_cluster_name)
    )

    assert np.array_equal(actual_cluster_layer, expected_output)
