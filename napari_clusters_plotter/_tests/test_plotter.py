import numpy as np
from skimage import measure

import napari_clusters_plotter as ncp
from napari_clusters_plotter._utilities import get_layer_tabular_data


def test_plotting(make_napari_viewer):

    viewer = make_napari_viewer()
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

    # image = label * 1.5

    props = measure.regionprops_table(
        label, properties=(["label", "area", "perimeter"])
    )

    label_layer = viewer.add_labels(label, properties=props)
    # image_layer = viewer.add_image(image)

    for widget in widget_list:
        _widget = widget(viewer)
        # Doesn't work for now to use in tests because MeasureWidget uses cle function
        # if isinstance(_widget, ncp._measure.MeasureWidget):
        #     _widget.run(
        #         image_layer, label_layer, "Measure now intensity shape", None, None
        #     )

        if isinstance(_widget, ncp._plotter.PlotterWidget):
            plot_widget = _widget

    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == 1
    # assert len(viewer.window._dock_widgets) == 2

    result = get_layer_tabular_data(label_layer)

    assert "label" in result.columns
    assert "area" in result.columns
    assert "perimeter" in result.columns


if __name__ == "__main__":
    test_plotting()
