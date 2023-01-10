import pytest

import napari_clusters_plotter

# import dask.array as da
# import numpy as np


# from napari_clusters_plotter._measure import get_regprops_from_regprops_source

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-clusters-plotter"
# the name of your widget(s)
MY_WIDGET_NAMES = [
    "Plotter Widget",
    "Dimensionality Reduction Widget",
    "Clustering Widget",
]

@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_widget_creation(widget_name, make_napari_viewer, napari_plugin_manager):
    """Function to test docking widgets into viewer"""
    napari_plugin_manager.register(napari_clusters_plotter, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1
