import numpy as np
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
        result[i] if frame != current_frame else highlight
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
