import napari

import napari_clusters_plotter as ncp


def test_widget_creation():

    viewer = napari.Viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    for widget in widget_list:

        _widget = widget(viewer)
        viewer.window.add_dock_widget(_widget)

    assert len(viewer.window._dock_widgets) == n_wdgts + 4


if __name__ == "__main__":
    test_widget_creation()
