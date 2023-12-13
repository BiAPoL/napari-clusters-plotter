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

    mapped_image_2d = map_array(label_image_2d, input_values, output_values)

    assert mapped_image_2d[0, 0] == 0
    assert mapped_image_2d[0, 1] == 1
    assert mapped_image_2d[0, 2] == 2

    assert mapped_image_2d[1, 0] == 1
    assert mapped_image_2d[1, 1] == 1
    assert mapped_image_2d[1, 2] == 1

    assert mapped_image_2d[2, 0] == 0
    assert mapped_image_2d[2, 1] == 1
    assert mapped_image_2d[2, 2] == 2


def test_map_array_4d():
    from skimage.util import map_array

    output_array = map_array(label_image_4d, input_values, output_values)

    assert output_array[0, 0, 0, 0] == 0
    assert output_array[0, 0, 0, 1] == 1
    assert output_array[0, 0, 0, 2] == 2
    assert output_array[0, 0, 1, 0] == 1
    assert output_array[0, 0, 1, 1] == 1
    assert output_array[0, 0, 1, 2] == 1
    assert output_array[0, 0, 2, 0] == 0
    assert output_array[0, 0, 2, 1] == 1
    assert output_array[0, 0, 2, 2] == 2

    assert output_array[0, 1, 0, 0] == 1
    assert output_array[0, 1, 0, 1] == 1
    assert output_array[0, 1, 0, 2] == 1
    assert output_array[0, 1, 1, 0] == 0
    assert output_array[0, 1, 1, 1] == 1
    assert output_array[0, 1, 1, 2] == 2
    assert output_array[0, 1, 2, 0] == 1
    assert output_array[0, 1, 2, 1] == 1
    assert output_array[0, 1, 2, 2] == 1

    assert output_array[1, 0, 0, 0] == 0
    assert output_array[1, 0, 0, 1] == 1
    assert output_array[1, 0, 0, 2] == 2
    assert output_array[1, 0, 1, 0] == 1
    assert output_array[1, 0, 1, 1] == 1
    assert output_array[1, 0, 1, 2] == 1
    assert output_array[1, 0, 2, 0] == 0
    assert output_array[1, 0, 2, 1] == 1
    assert output_array[1, 0, 2, 2] == 2

    assert output_array[1, 1, 0, 0] == 1
    assert output_array[1, 1, 0, 1] == 1
    assert output_array[1, 1, 0, 2] == 1
    assert output_array[1, 1, 1, 0] == 0
    assert output_array[1, 1, 1, 1] == 1
    assert output_array[1, 1, 1, 2] == 2
    assert output_array[1, 1, 2, 0] == 1
    assert output_array[1, 1, 2, 1] == 1
    assert output_array[1, 1, 2, 2] == 1


def test_generate_cluster_image():
    input_labels = [1, 2, 3, 4, 5]
    output_labels = [1, 1, 0, 0, 0]

    cluster_label_image_3d = utilities.generate_cluster_image(
        label_image_3d, input_labels, output_labels
    )
    cluster_label_image_4d = utilities.generate_cluster_image(
        label_image_4d, input_labels, output_labels
    )

    assert cluster_label_image_3d[0, 0, 0] == 0
    assert cluster_label_image_3d[0, 0, 1] == 2
    assert cluster_label_image_3d[0, 0, 2] == 2
    assert cluster_label_image_3d[0, 1, 0] == 1
    assert cluster_label_image_3d[0, 1, 1] == 1
    assert cluster_label_image_3d[0, 1, 2] == 1

    assert cluster_label_image_4d[0, 0, 0, 0] == 0
    assert cluster_label_image_4d[0, 0, 0, 1] == 2
    assert cluster_label_image_4d[0, 0, 0, 2] == 2
    assert cluster_label_image_4d[0, 0, 1, 0] == 1
    assert cluster_label_image_4d[0, 0, 1, 1] == 1
    assert cluster_label_image_4d[0, 0, 1, 2] == 1
    assert cluster_label_image_4d[0, 0, 2, 0] == 0
    assert cluster_label_image_4d[0, 0, 2, 1] == 2
    assert cluster_label_image_4d[0, 0, 2, 2] == 2


def test_generate_cluster_image_ocl_array():
    input_labels = [1, 2, 3, 4, 5]
    output_labels = [1, 1, 0, 0, 0]

    import pyclesperanto_prototype._tier0._pycl as pycl

    ocl = pycl.OCLArray.from_array(label_image_2d)

    cluster_label_image_2d = utilities.generate_cluster_image(
        ocl, input_labels, output_labels
    )

    assert cluster_label_image_2d[0, 0] == 0
    assert cluster_label_image_2d[0, 1] == 2
    assert cluster_label_image_2d[0, 2] == 2


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
    cluster_layer = utilities.generate_cluster_tracks(labels_layer, plot_cluster_name)

    assert cluster_layer[0, 0, 0, 0] == 0
    assert cluster_layer[0, 0, 0, 1] == 2
    assert cluster_layer[0, 0, 0, 2] == 2
    assert cluster_layer[0, 0, 1, 0] == 1
    assert cluster_layer[0, 0, 1, 1] == 1
    assert cluster_layer[0, 0, 1, 2] == 1
    assert cluster_layer[0, 0, 2, 0] == 0
    assert cluster_layer[0, 0, 2, 1] == 2
    assert cluster_layer[0, 0, 2, 2] == 2
