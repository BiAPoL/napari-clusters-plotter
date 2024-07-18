import numpy as np
import pandas as pd

from napari_clusters_plotter._plotter import PlotterWidget


def create_fake_surface():
    vertices = np.array([[0, 0], [0, 20], [10, 0], [10, 10]])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    values = np.linspace(0, 1, len(vertices))
    return vertices, faces, values


def get_fake_surface_measurements():
    measurements = {
        "vertex_index": {0: 0, 1: 1, 2: 2, 3: 3},
        "Quality.ASPECT_RATIO": {
            0: 1.6899336730388865,
            1: 2.345838749778065,
            2: 2.345838749778065,
            3: 3.001743826517244,
        },
        "Quality.AREA": {0: 100.0, 1: 75.0, 2: 75.0, 3: 50.0},
    }
    return pd.DataFrame.from_dict(measurements)


def test_surface_data_plotting(make_napari_viewer):
    viewer = make_napari_viewer()

    surface = create_fake_surface()
    measurements = get_fake_surface_measurements()

    surface_layer = viewer.add_surface(surface)
    surface_layer.features = measurements

    viewer.window.add_dock_widget(PlotterWidget(viewer), area="right")
    plotter_widget = PlotterWidget(viewer)

    plotter_widget.run(
        features=measurements,
        plot_x_axis_name="Quality.ASPECT_RATIO",
        plot_y_axis_name="Quality.AREA",
        redraw_cluster_image=True,
        force_redraw=True,
    )

    # check if plot has data
    assert plotter_widget.graphics_widget.axes.has_data()


def test_dimensionality_reduction_for_surface_data(make_napari_viewer):
    from napari_clusters_plotter._dimensionality_reduction import (
        DimensionalityReductionWidget,
    )

    viewer = make_napari_viewer()
    surface = create_fake_surface()
    measurements = get_fake_surface_measurements()

    surface_layer = viewer.add_surface(surface)
    surface_layer.features = measurements

    widget = DimensionalityReductionWidget(napari_viewer=viewer)
    widget.run(
        viewer=viewer,
        layer=surface_layer,
        selected_measurements_list=list(measurements.keys()),
        n_neighbours=2,
        perplexity=5,
        selected_algorithm="UMAP",
        standardize=False,
        n_components=2,
        explained_variance=95.0,
        pca_components=0,
        mds_metric=True,
        mds_n_init=4,
        mds_max_iter=300,
        mds_eps=0.001,
        umap_multithreading=False,
        min_dist=0.1,
        custom_name="",
    )

    assert "UMAP_0" in list(surface_layer.features.keys())
    assert "UMAP_1" in list(surface_layer.features.keys())


def test_cluster_ids_layer_generation_for_surface_data(make_napari_viewer):
    from napari_clusters_plotter._plotter import PlotterWidget

    viewer = make_napari_viewer()
    surface = create_fake_surface()
    measurements = get_fake_surface_measurements()

    measurements["MANUAL_CLUSTER_ID"] = 1
    measurements["MANUAL_CLUSTER_ID"].loc[1] = 2
    measurements["MANUAL_CLUSTER_ID"].loc[3] = 2

    surface_layer = viewer.add_surface(surface)
    surface_layer.features = measurements

    viewer.window.add_dock_widget(PlotterWidget(viewer), area="right")
    plotter_widget = PlotterWidget(viewer)

    plotter_widget.run(
        features=measurements,
        plot_x_axis_name="Quality.ASPECT_RATIO",
        plot_y_axis_name="Quality.AREA",
        plot_cluster_name="MANUAL_CLUSTER_ID",
        redraw_cluster_image=True,
        force_redraw=True,
    )

    assert plotter_widget.graphics_widget.axes.has_data()
    assert len(viewer.layers) == 2
    assert "cluster_ids_in_space" in viewer.layers
