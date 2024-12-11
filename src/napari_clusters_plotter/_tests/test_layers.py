import numpy as np
import pandas as pd

import napari_clusters_plotter as ncp


def test_shapes_layer(make_napari_viewer):

    # random centers
    box_centers = np.random.random((20, 2)) * 1000
    box_widths = np.random.random(20) * 100
    box_heights = np.random.random(20) * 100
    features = pd.DataFrame(
        {"box_widths": box_widths, "box_heights": box_heights}
    )
    features["area"] = features["box_widths"] * features["box_heights"]

    boxes = []

    for i in range(20):
        corners = np.array(
            [
                [
                    box_centers[i, 0] - box_widths[i] / 2,
                    box_centers[i, 1] - box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] + box_widths[i] / 2,
                    box_centers[i, 1] - box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] + box_widths[i] / 2,
                    box_centers[i, 1] + box_heights[i] / 2,
                ],
                [
                    box_centers[i, 0] - box_widths[i] / 2,
                    box_centers[i, 1] + box_heights[i] / 2,
                ],
            ]
        )
        boxes.append(corners)

    viewer = make_napari_viewer()
    viewer.add_shapes(boxes, shape_type="polygon", features=features)

    plotter_widget = ncp.PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter_widget, area="right")
