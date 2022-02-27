import napari
import numpy as np

import napari_clusters_plotter as ncp


def test_measurements():

    viewer = napari.Viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()

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

    image = label * 1.5

    label_layer = viewer.add_labels(label)
    image_layer = viewer.add_image(image)

    for widget in widget_list:
        _widget = widget(viewer)
        if isinstance(_widget, ncp.MeasureWidget):
            break

    viewer.window.add_dock_widget(_widget)

    _widget.run(image_layer, label_layer, "Measure now intensity", None, None)
    data = label_layer.features
    assert "max_intensity" in data.columns
    assert "sum_intensity" in data.columns
    assert "mean_intensity" in data.columns
    assert "min_intensity" in data.columns

    assert data["max_intensity"].max() == 7 * 1.5

    _widget.run(image_layer, label_layer, "Measure now shape", None, None)
    data = label_layer.features
    assert "area" in data.columns
    assert "mean_distance_to_centroid" in data.columns
    assert "max_distance_to_centroid" in data.columns
    assert "mean_max_distance_to_centroid_ratio" in data.columns

    assert data["area"].max() == 5

    _widget.run(image_layer, label_layer, "Measure now neighborhood", None, "2, 3, 4")
    data = label_layer.features
    assert "avg distance of 2 closest points" in data.columns
    assert "avg distance of 3 closest points" in data.columns
    assert "avg distance of 4 closest points" in data.columns
    assert "touching neighbor count" in data.columns
    assert data["touching neighbor count"].loc[5] == 2


if __name__ == "__main__":
    test_measurements()
