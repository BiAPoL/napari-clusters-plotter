import numpy as np
import pandas as pd
from skimage import measure

import napari_clusters_plotter as ncp
from napari_clusters_plotter._plotter_utilities import (
    alpha_factor,
    alphas_clustered,
    alphas_unclustered,
    clustered_plot_parameters,
    colors_clustered,
    colors_unclustered,
    frame_spot_factor,
    gen_highlight,
    gen_spot_size,
    initial_and_noise_alpha,
    spot_size_clustered,
    spot_size_unclustered,
    unclustered_plot_parameters,
)
from napari_clusters_plotter._utilities import get_layer_tabular_data, get_nice_colormap


def get_labels_array():
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
    return label


def test_plotting(make_napari_viewer):
    viewer = make_napari_viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()

    label = get_labels_array()
    annotations = np.random.randint(0, 2, size=label.max())

    props = measure.regionprops_table(
        label, properties=(["label", "area", "perimeter"])
    )

    label_layer = viewer.add_labels(label, properties=props)

    for widget in widget_list:
        _widget = widget(viewer)

        if isinstance(_widget, ncp._plotter.PlotterWidget):
            plot_widget = _widget

    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == 1

    result = get_layer_tabular_data(label_layer)

    assert "label" in result.columns
    assert "area" in result.columns
    assert "perimeter" in result.columns

    plot_widget.analysed_layer = viewer.layers[0]
    plot_widget.plot_x_axis.setCurrentText("area")
    plot_widget.plot_y_axis.setCurrentText("perimeter")
    plot_widget.analysed_layer.features["MANUAL_CLUSTER_ID"] = annotations

    plot_widget.run(
        plot_widget.analysed_layer.features,
        plot_x_axis_name="area",
        plot_y_axis_name="perimeter",
        plot_cluster_name="MANUAL_CLUSTER_ID",
    )

    print("check")


def test_plotter_utilities():
    frame_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    predicts = [0, 1, -1, 1, 0, 1, 1, 0]
    n_datapoints = len(predicts)
    current_frame = 2

    frame_spot_f = frame_spot_factor()
    init_alpha, noise_alpha = initial_and_noise_alpha()
    alpha_f = alpha_factor(n_datapoints)
    spot_size = gen_spot_size(n_datapoints)

    colors = get_nice_colormap()
    highlight = gen_highlight()

    alpha_clustered = alphas_clustered(predicts, frame_ids, current_frame, n_datapoints)

    result = [
        alpha_f * init_alpha * 0.3 if pred >= 0 else alpha_f * noise_alpha * 0.3
        for pred in predicts
    ]
    result_ac = [
        result[i] if frame != current_frame else alpha_f * init_alpha
        for i, frame in enumerate(frame_ids)
    ]

    assert alpha_clustered == result_ac

    alpha_unclustered = alphas_unclustered(frame_ids, current_frame, n_datapoints)
    result_au = [
        alpha_f * init_alpha * 0.3 if frame != current_frame else alpha_f * init_alpha
        for frame in frame_ids
    ]

    assert alpha_unclustered == result_au

    spots_clustered = spot_size_clustered(
        predicts, frame_ids, current_frame, n_datapoints
    )
    result = [spot_size if pred >= 0 else spot_size / 2 for pred in predicts]
    result_sc = [
        result[i] if frame != current_frame else result[i] * frame_spot_f
        for i, frame in enumerate(frame_ids)
    ]

    assert spots_clustered == result_sc

    spots_unclustered = spot_size_unclustered(frame_ids, current_frame, n_datapoints)
    result_su = [
        spot_size if frame != current_frame else spot_size * frame_spot_f
        for frame in frame_ids
    ]

    assert spots_unclustered == result_su

    colors_cl = colors_clustered(predicts, frame_ids, current_frame, colors)
    result = [colors[pred] if pred >= 0 else "#bcbcbc" for pred in predicts]
    result_cc = [
        result[i] if frame != current_frame else gen_highlight(result[i])
        for i, frame in enumerate(frame_ids)
    ]

    assert colors_cl == result_cc

    colors_uc = colors_unclustered(frame_ids, current_frame)
    result_cu = [
        "#9A9A9A" if frame != current_frame else highlight
        for i, frame in enumerate(frame_ids)
    ]

    assert colors_uc == result_cu

    cl_plot_params = clustered_plot_parameters(
        predicts, frame_ids, current_frame, n_datapoints, colors
    )
    assert cl_plot_params == (result_ac, result_sc, result_cc)

    uc_plot_params = unclustered_plot_parameters(frame_ids, current_frame, n_datapoints)
    assert uc_plot_params == (result_au, result_su, result_cu)


def test_plotting_histogram(make_napari_viewer):
    from napari_clusters_plotter._plotter import PlotterWidget

    viewer = make_napari_viewer()

    label = get_labels_array()
    measurements = measure.regionprops_table(
        label, properties=(["label", "area", "perimeter"])
    )
    label_layer = viewer.add_labels(label, properties=measurements)
    label_layer.features = measurements

    viewer.window.add_dock_widget(PlotterWidget(viewer), area="right")
    plotter_widget = PlotterWidget(viewer)
    plotter_widget.plotting_type.setCurrentText("HISTOGRAM_2D")

    plotter_widget.run(
        features=pd.DataFrame(measurements),
        plot_x_axis_name="area",
        plot_y_axis_name="perimeter",
        force_redraw=True,
    )

    assert plotter_widget.graphics_widget.axes.has_data()

    # test plotting 1D histogram
    plotter_widget.run(
        features=pd.DataFrame(measurements),
        plot_x_axis_name="area",
        plot_y_axis_name="area",
        force_redraw=True,
    )

    assert plotter_widget.graphics_widget.axes.has_data()


def test_plotter_on_points_data(make_napari_viewer):
    from napari_clusters_plotter._plotter import PlotterWidget
    from napari_clusters_plotter._utilities import get_layer_tabular_data

    viewer = make_napari_viewer()

    n_points = 100
    points = np.random.rand(n_points, 2)
    feature1 = np.random.normal(size=n_points)
    feature2 = np.random.normal(size=n_points, loc=1)
    annotations = 1 * (np.random.uniform(size=100) > 0.5)

    viewer.add_points(points, properties={"feature1": feature1, "feature2": feature2})

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget)

    # Put the points layer into the layer selection widget
    widget.analysed_layer = viewer.layers[0]
    widget.plot_x_axis.setCurrentText("feature1")
    widget.plot_y_axis.setCurrentText("feature2")
    widget.analysed_layer.features["MANUAL_CLUSTER_ID"] = annotations

    # check that the features are found
    features = get_layer_tabular_data(widget.analysed_layer)
    assert "feature1" in features.columns
    assert "feature2" in features.columns

    widget.run(
        widget.analysed_layer.features,
        plot_x_axis_name="feature1",
        plot_y_axis_name="feature2",
        plot_cluster_name="MANUAL_CLUSTER_ID",
        force_redraw=True,
    )


def test_plotter_on_surface_data(make_napari_viewer):
    from napari_clusters_plotter._plotter import PlotterWidget
    from napari_clusters_plotter._utilities import get_layer_tabular_data

    viewer = make_napari_viewer()

    # Create a random mesh
    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, (100, 3))

    feature1 = np.random.normal(size=100)
    feature2 = np.random.normal(size=100, loc=1)
    annotations = 1 * (np.random.uniform(size=100) > 0.5)

    layer = viewer.add_surface((vertices, faces))
    layer.features = pd.DataFrame({"feature1": feature1, "feature2": feature2})

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    # Put the surface layer into the layer selection widget
    widget.analysed_layer = viewer.layers[0]
    widget.plot_x_axis.setCurrentText("feature1")
    widget.plot_y_axis.setCurrentText("feature2")
    widget.analysed_layer.features["MANUAL_CLUSTER_ID"] = annotations

    # check that the features are found
    features = get_layer_tabular_data(widget.analysed_layer)
    assert "feature1" in features.columns
    assert "feature2" in features.columns

    widget.run(
        widget.analysed_layer.features,
        plot_x_axis_name="feature1",
        plot_y_axis_name="feature2",
        plot_cluster_name="MANUAL_CLUSTER_ID",
        force_redraw=True,
    )


def test_cluster_image_generation_for_histogram(make_napari_viewer):
    from napari_clusters_plotter._plotter import PlotterWidget

    viewer = make_napari_viewer()

    label = get_labels_array()
    measurements = measure.regionprops_table(
        label, properties=(["label", "area", "perimeter"])
    )
    measurements["MANUAL_CLUSTER_ID"] = np.array([1, 0, 2, -1, 0, 1, 2])
    viewer.add_labels(label, properties=measurements)

    viewer.window.add_dock_widget(PlotterWidget(viewer), area="right")
    plotter_widget = PlotterWidget(viewer)
    plotter_widget.plotting_type.setCurrentText("HISTOGRAM_2D")
    plotter_widget.log_scale.value = True

    plotter_widget.run(
        features=pd.DataFrame(measurements),
        plot_x_axis_name="area",
        plot_y_axis_name="perimeter",
        plot_cluster_name="MANUAL_CLUSTER_ID",
        force_redraw=True,
    )

    assert plotter_widget.graphics_widget.axes.has_data()
    assert "cluster_ids_in_space" in viewer.layers
    assert int(viewer.layers["cluster_ids_in_space"].data.max()) == 3
