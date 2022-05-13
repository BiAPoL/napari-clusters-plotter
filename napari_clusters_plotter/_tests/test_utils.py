import sys

import numpy as np
import pandas as pd
from numpy import array

from napari_clusters_plotter._utilities import (
    add_column_to_layer_tabular_data,
    dask_cluster_image_timelapse,
    generate_cluster_image,
    generate_cmap_dict,
    generate_label_to_cluster_color_mapping,
    get_layer_tabular_data,
    get_nice_colormap,
    reshape_2D_timelapse,
    set_features,
)

sys.path.append("../")


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


def test_colormaps_and_mappings():
    colors = get_nice_colormap()
    assert colors[3] == "#d62728"

    predictions = [0, 1, 0, 0, 2, 0]

    labels = [1, 2, 3, 4, 5, 6]

    cmap_dict = generate_cmap_dict(colors=colors, prediction_list=predictions)

    result_cmap = {
        0: array([0.0, 0.0, 0.0, 0.0]),
        1: array([1.0, 0.49803922, 0.05490196, 1.0]),
        2: array([0.12156863, 0.46666667, 0.70588235, 1.0]),
    }

    for i in cmap_dict.keys():
        assert np.allclose(cmap_dict[i], result_cmap[i])

    mapping = generate_label_to_cluster_color_mapping(
        label_list=labels,
        predictionlist=np.asarray(predictions) - 1,
        colormap_dict=cmap_dict,
    )
    mapping_result = {
        0: array([0.0, 0.0, 0.0, 0.0]),
        1: array([0.0, 0.0, 0.0, 0.0]),
        2: array([1.0, 0.49803922, 0.05490196, 1.0]),
        3: array([0.0, 0.0, 0.0, 0.0]),
        4: array([0.0, 0.0, 0.0, 0.0]),
        5: array([0.12156863, 0.46666667, 0.70588235, 1.0]),
        6: array([0.0, 0.0, 0.0, 0.0]),
    }

    for key in mapping.keys():
        assert np.allclose(mapping[key], mapping_result[key])


def test_old_functions():
    label_1 = np.array(
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

    label_2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 0],
        ]
    )

    time_lapse_2d = np.array([label_1, label_2])

    reshaped_time_lapse = reshape_2D_timelapse(time_lapse_2d)

    assert reshaped_time_lapse.shape == (2, 1, 7, 7)
    predictions = [[0, 0, 1, 0, 1, 0, 0], [0, 1]]

    # just call them
    generate_cluster_image(time_lapse_2d[0],predictions[0])
    dask_cluster_image_timelapse(reshaped_time_lapse,predictions)


if __name__ == "__main__":
    import napari

    test_feature_setting(napari.Viewer)
