import pytest
import numpy as np
import pandas as pd


def create_points(n_samples=20, loc=5):
    from napari.layers import Points

    points = np.random.random((n_samples, 3))

    features = pd.DataFrame({'feature1': np.random.normal(size=n_samples, loc=loc),
                         'feature2': np.random.normal(size=n_samples, loc=loc),
                         'feature3': np.random.normal(size=n_samples, loc=loc),
                         'feature4': np.random.normal(size=n_samples, loc=loc),})
    
    layer = Points(points, features=features, size=0.1)

    return layer

def test_initialization(make_napari_viewer):
    from napari_clusters_plotter import DimensionalityReductionWidget
    
    viewer = make_napari_viewer()
    layer = create_points()
    viewer.add_layer(layer)

    count_widgets = len(viewer.window._dock_widgets)
    reducer_widget = DimensionalityReductionWidget(viewer)
    viewer.window.add_dock_widget(reducer_widget, area="right")

    assert reducer_widget.viewer == viewer
    assert len(viewer.window._dock_widgets) == count_widgets + 1
    for algorithm in ["PCA", "t-SNE", "UMAP"]:
        assert algorithm in reducer_widget.algorithms

    # check if the widget is properly initialized
    feature_selection_items = [
        reducer_widget.feature_selection_widget.item(i).text()
        for i in range(reducer_widget.feature_selection_widget.count())
        ]
    
    for feature in layer.features.columns:
        assert feature in reducer_widget.common_columns
        assert feature in feature_selection_items
        
    # check that all features are in reduce_wdiget.feature_selection_widget
    assert len(feature_selection_items) == len(layer.features.columns)

def test_layer_update(make_napari_viewer):
    from napari_clusters_plotter import DimensionalityReductionWidget
    
    viewer = make_napari_viewer()
    # Create two random layers using the create_points fixture
    points_layer1 = create_points()
    points_layer2 = create_points()
    
    points_layer1.name = "points1"
    points_layer2.name = "points2"
    
    # remove one feature from the second layer
    points_layer2.features = points_layer2.features.drop(columns=["feature4"])

    # Add the layers to the viewer
    viewer.add_layer(points_layer1)
    viewer.add_layer(points_layer2)

    reducer_widget = DimensionalityReductionWidget(viewer)
    viewer.window.add_dock_widget(reducer_widget, area="right")

    selection_permutations = [
        [points_layer1],
        [points_layer2],
        [points_layer1, points_layer2]
    ]

    for possible_selection in selection_permutations:
        viewer.layers.selection = possible_selection

        feature_selection_items = [
            reducer_widget.feature_selection_widget.item(i).text()
            for i in range(reducer_widget.feature_selection_widget.count())
            ]
        
        for feature in reducer_widget.common_columns:
            # make sure common columns are in every layer
            assert feature in feature_selection_items
            for layer in possible_selection:
                assert feature in layer.features.columns

        # make sure that the name of the layer is in the collected features
        collected_features = reducer_widget._get_features()
        for layer in possible_selection:
            for feature in reducer_widget.common_columns:
                assert feature in collected_features.columns
                assert layer.name in collected_features["layer"].values

def test_feature_update(make_napari_viewer):
    from napari_clusters_plotter import DimensionalityReductionWidget
    viewer = make_napari_viewer()

    points_layer = create_points()
    viewer.add_layer(points_layer)

    reducer_widget = DimensionalityReductionWidget(viewer)
    viewer.window.add_dock_widget(reducer_widget, area="right")

    # select all possible permutations of features
    combinations = [
        [0],
        [0, 1],
        [0, 1, 2],
    ]

    for combination in combinations:
        for idx in combination:
            # select the feature
            item = reducer_widget.feature_selection_widget.item(idx)
            item.setSelected(True)

            features = reducer_widget._update_features()
            selected_features = [item.text() for item in reducer_widget.feature_selection_widget.selectedItems()]
            assert all([feature in selected_features for feature in features.columns])

            for selected_feature in selected_features:
                assert selected_feature in features.columns
                assert selected_feature in points_layer.features.columns
                assert selected_feature in reducer_widget.selected_algorithm_widget.data.value.columns


def test_algorithm_change(make_napari_viewer):
    from napari_clusters_plotter import DimensionalityReductionWidget
    viewer = make_napari_viewer()

    points_layer = create_points()
    viewer.add_layer(points_layer)

    reducer_widget = DimensionalityReductionWidget(viewer)
    viewer.window.add_dock_widget(reducer_widget, area="right")

    # select all possible permutations of features
    for idx in range(reducer_widget.algorithm_selection.count()):
        reducer_widget.algorithm_selection.setCurrentIndex(idx)
        assert reducer_widget.selected_algorithm == reducer_widget.algorithm_selection.itemText(idx)

