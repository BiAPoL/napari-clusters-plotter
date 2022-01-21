# -*- coding: utf-8 -*-


import napari_clusters_plotter as ncp
import napari

import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from _utilities import set_features, get_layer_tabular_data, add_column_to_layer_tabular_data

def test_feature_setting():
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
    label_layer = viewer.add_labels(label)

    some_features = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    set_features(label_layer, some_features)

    assert isinstance(label_layer.features, pd.DataFrame)

    some_features = get_layer_tabular_data(label_layer)
    assert isinstance(some_features, pd.DataFrame)

    add_column_to_layer_tabular_data(label_layer, 'C', [5, 6, 7])
    some_features = get_layer_tabular_data(label_layer)
    assert 'C' in some_features.columns


if __name__ == '__main__':
    test_feature_setting()