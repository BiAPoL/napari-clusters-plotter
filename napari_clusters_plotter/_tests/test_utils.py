import sys

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QListWidget

from napari_clusters_plotter._utilities import (
    add_column_to_layer_tabular_data,
    dask_cluster_image_timelapse,
    generate_cluster_image,
    get_layer_tabular_data,
    reshape_2D_timelapse,
    set_features,
    update_properties_list,
)

sys.path.append("../")


class FakeWidget:
    def __init__(self, layer):
        class Labels_select:
            def __init__(self, layer):
                self.value = layer

        self.properties_list = QListWidget()
        self.labels_select = Labels_select(layer)


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
    result = generate_cluster_image(label, label_ids, predictions)
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
    label_timelapse = reshape_2D_timelapse(label_timelapse_3d)

    label_id_list = np.array([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])
    predictions_list = np.array([[0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1, 0]])
    result_dask = dask_cluster_image_timelapse(label_timelapse, label_id_list, predictions_list)
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
    set_features(label_layer, some_features)

    if hasattr(label_layer, "features"):
        assert isinstance(label_layer.features, pd.DataFrame)
    elif hasattr(label_layer, "properties"):
        assert isinstance(label_layer.properties, dict)

    some_features = get_layer_tabular_data(label_layer)
    assert isinstance(some_features, pd.DataFrame)

    add_column_to_layer_tabular_data(label_layer, "C", [5, 6, 7])
    some_features = get_layer_tabular_data(label_layer)
    assert "C" in some_features.columns

    widget = FakeWidget(label_layer)
    update_properties_list(widget, exclude_list=["A"])
