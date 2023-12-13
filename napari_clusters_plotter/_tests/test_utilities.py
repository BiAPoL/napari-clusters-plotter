import napari
import numpy as np

import napari_clusters_plotter._utilities as utilities

input_values = np.array([0, 1, 2, 3, 4, 5])
output_values = np.array([0, 1, 2, 1, 1, 1])

label_image_2d_float = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])

label_image_2d = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]])

label_image_3d = np.array(
    [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]]
)

label_image_4d = np.array(
    [
        [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]],
        [[[0, 1, 2], [3, 4, 5], [0, 1, 2]], [[3, 4, 5], [0, 1, 2], [3, 4, 5]]],
    ]
)


def test_map_array_3d():
    from skimage.util import map_array

    actual_mapped_image_2d = map_array(label_image_2d, input_values, output_values)
    expected_mapped_image_2d = np.array([[0, 1, 2], [1, 1, 1], [0, 1, 2]])

    assert np.array_equal(actual_mapped_image_2d, expected_mapped_image_2d)


def test_map_array_4d():
    from skimage.util import map_array

    actual_output_array = map_array(label_image_4d, input_values, output_values)
    expected_output_array = np.array(
        [
            [[[0, 1, 2], [1, 1, 1], [0, 1, 2]], [[1, 1, 1], [0, 1, 2], [1, 1, 1]]],
            [[[0, 1, 2], [1, 1, 1], [0, 1, 2]], [[1, 1, 1], [0, 1, 2], [1, 1, 1]]],
        ]
    )

    assert np.array_equal(actual_output_array, expected_output_array)


def test_generate_cluster_image():
    input_labels = [1, 2, 3, 4, 5]
    output_labels = [1, 1, 0, 0, 0]

    actual_cluster_label_image_3d = utilities.generate_cluster_image(
        label_image_3d, input_labels, output_labels
    )
    expected_cluster_label_image_3d = np.array(
        [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]]
    )

    actual_cluster_label_image_4d = utilities.generate_cluster_image(
        label_image_4d, input_labels, output_labels
    )
    expected_cluster_label_image_4d = np.array(
        [
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
        ]
    )

    assert np.array_equal(
        actual_cluster_label_image_3d, expected_cluster_label_image_3d
    )
    assert np.array_equal(
        actual_cluster_label_image_4d, expected_cluster_label_image_4d
    )


def test_generate_cluster_image_ocl_array():
    input_labels = [1, 2, 3, 4, 5]
    output_labels = [1, 1, 0, 0, 0]

    import pyclesperanto_prototype._tier0._pycl as pycl

    ocl = pycl.OCLArray.from_array(label_image_2d)

    actual_cluster_label_image_2d = utilities.generate_cluster_image(
        ocl, input_labels, output_labels
    )
    expected_cluster_label_image_2d = np.array([[0, 2, 2], [1, 1, 1], [0, 2, 2]])

    assert np.array_equal(
        actual_cluster_label_image_2d, expected_cluster_label_image_2d
    )


def test_generate_cluster_tracks():
    plot_cluster_name = "MANUAL_CLUSTER_ID"

    import pandas as pd

    # Creating a DataFrame from a dictionary
    features = {
        "label": [1, 2, 3, 4, 5],
        "MANUAL_CLUSTER_ID": [1, 1, 0, 0, 0],
        "value": [2.0, 3.0, 42.0, 50.0, 60.0],
    }

    features = pd.DataFrame(features)
    labels_layer = napari.layers.Labels(data=label_image_4d, features=features)
    actual_cluster_layer = utilities.generate_cluster_tracks(
        labels_layer, plot_cluster_name
    )
    expected_cluster_layer = np.array(
        [
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
            [[[0, 2, 2], [1, 1, 1], [0, 2, 2]], [[1, 1, 1], [0, 2, 2], [1, 1, 1]]],
        ]
    )

    assert np.array_equal(actual_cluster_layer, expected_cluster_layer)
