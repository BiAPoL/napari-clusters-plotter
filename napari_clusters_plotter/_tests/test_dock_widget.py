import dask.array as da
import numpy as np

import napari_clusters_plotter
from napari_clusters_plotter._measure import get_regprops_from_regprops_source

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-clusters-plotter"
# the name of your widget(s)
MY_WIDGET_NAMES = [
    "Measure Widget",
    "Plotter Widget",
    "Dimensionality Reduction Widget",
    "Clustering Widget",
]


def test_processing_dask_array(make_napari_viewer):
    """Function to test processing dask arrays"""
    # Create generic dask image and label_image inputs
    data = np.arange(100).reshape(4, 25)
    dask_image = da.from_array(data, chunks=(10, 10))
    label_image = np.zeros_like(data)
    label_image[:, 5:10] = 1
    label_image[:, 10:15] = 2
    label_image[1:, 16:20] = 3
    dask_label_image = da.from_array(label_image, chunks=(10, 10))
    region_props_source = "Measure now (with neighborhood data)"

    # Expected region_props output
    expected_region_props_partial = {
        "label": np.array([1, 2, 3]),
        "area": np.array([20.0, 20.0, 12.0]),
        "mean_intensity": np.array([44.5, 49.5, 67.5]),
    }
    viewer = make_napari_viewer()
    img_layer = viewer.add_image(dask_image)
    label_layer = viewer.add_labels(dask_label_image)
    region_props = get_regprops_from_regprops_source(
        img_layer.data, label_layer.data, region_props_source
    )
    assert np.all(
        np.array_equal(expected_region_props_partial["label"], region_props["label"])
        and np.array_equal(expected_region_props_partial["area"], region_props["area"])
        and np.array_equal(
            expected_region_props_partial["mean_intensity"],
            region_props["mean_intensity"],
        )
    )


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
