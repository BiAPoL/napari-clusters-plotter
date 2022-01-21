# -*- coding: utf-8 -*-
import napari_clusters_plotter as ncp
import napari
import numpy as np

import sys
sys.path.append('../')

from _utilities import get_layer_tabular_data


def test_plotting():

    viewer = napari.Viewer()
    widget_list = ncp.napari_experimental_provide_dock_widget()
    n_wdgts = len(viewer.window._dock_widgets)

    label = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 2, 2],
                      [0, 0, 0, 0, 2, 2, 2],
                      [3, 3, 0, 0, 0, 0, 0],
                      [0, 0, 4, 4, 0, 5, 5],
                      [6, 6, 6, 6, 0, 5, 0],
                      [0, 7, 7, 0, 0, 0, 0]])

    image = label * 1.5

    label_layer = viewer.add_labels(label)
    image_layer = viewer.add_image(image)

    for widget in widget_list:
        _widget = widget(viewer)
        if isinstance(_widget, ncp.MeasureWidget):
            _widget.run(image_layer, label_layer,
                        'Measure now intensity shape', None, None)

        if isinstance(_widget, ncp.PlotterWidget):
            plot_widget = _widget

    features = get_layer_tabular_data(label_layer)
    viewer.window.add_dock_widget(plot_widget)
    assert len(viewer.window._dock_widgets) == 2

    # redraw the whole plot
    plot_widget.viewer = viewer
    plot_widget.analysed_layer = [label_layer]
    plot_widget.plot_x_axis_name = 'max_intensity'
    plot_widget.plot_y_axis_name = 'mean_intensity'

    print(plot_widget.labels_select.value)


    # print(plot_widget.analysed_layer)
    # plot_widget.run(features, plot_widget.plot_x_axis_name, plot_widget.plot_y_axis_name,
    #                 plot_cluster_name="MANUAL_CLUSTER_ID")


if __name__ == '__main__':
    test_plotting()