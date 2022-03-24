import numpy as np
import napari_clusters_plotter as ncp


def test_measurements(make_napari_viewer):

    viewer = make_napari_viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()

    label = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2, 2],
            [3, 3, 0, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0],
            [6, 6, 6, 6, 0, 5, 0],  # <-single pixel label
            [0, 7, 7, 0, 0, 0, 0],
        ]
    )

    image = np.random.random((label.shape))

    for widget in widget_list:
        _widget = widget(viewer)
        if isinstance(_widget, ncp._measure.MeasureWidget):
            break

    viewer.window.add_dock_widget(_widget)

    from napari_clusters_plotter._measure import get_regprops_from_regprops_source

    measurements = get_regprops_from_regprops_source(image, label, 'shape')
    measurements = get_regprops_from_regprops_source(image, label, 'intensity')
    measurements = get_regprops_from_regprops_source(image, label, 'shape + intensity')
    measurements = get_regprops_from_regprops_source(image, label, 'neigborhood')

if __name__ == '__main__':
    import napari
    test_measurements(napari.Viewer)
