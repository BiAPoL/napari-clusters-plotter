import numpy as np
import pandas as pd
import pytest

from napari_clusters_plotter import (
    ClusteringWidget,
    DimensionalityReductionWidget,
)


@pytest.fixture(
    params=[
        {
            "widget_class": DimensionalityReductionWidget,
            "algorithms": ["PCA", "t-SNE", "UMAP"],
        },
        {
            "widget_class": ClusteringWidget,
            "algorithms": ["KMeans", "HDBSCAN"],
        },
    ]
)
def widget_config(request):
    return request.param


def create_points(n_samples=100, loc=5):
    from napari.layers import Points

    points = np.random.random((n_samples, 3))

    features = pd.DataFrame(
        {
            "feature1": np.random.normal(size=n_samples, loc=loc),
            "feature2": np.random.normal(size=n_samples, loc=loc),
            "feature3": np.random.normal(size=n_samples, loc=loc),
            "feature4": np.random.normal(size=n_samples, loc=loc),
        }
    )

    # add some NaNs
    features.iloc[::10] = np.nan

    layer = Points(points, features=features, size=0.1)

    return layer


def test_initialization(make_napari_viewer, widget_config):
    viewer = make_napari_viewer()
    layer = create_points()
    viewer.add_layer(layer)

    count_widgets = len(viewer.window._dock_widgets)

    WidgetClass = widget_config["widget_class"]
    algorithms = widget_config["algorithms"]
    widget = WidgetClass(viewer)

    viewer.window.add_dock_widget(widget, area="right")

    assert widget.viewer == viewer
    assert len(viewer.window._dock_widgets) == count_widgets + 1
    for algorithm in algorithms:
        assert algorithm in widget.algorithms

    # check if the widget is properly initialized
    feature_selection_items = [
        widget.feature_selection_widget.item(i).text()
        for i in range(widget.feature_selection_widget.count())
    ]

    for feature in layer.features.columns:
        assert feature in widget.common_columns
        assert feature in feature_selection_items

    # check that all features are in reduce_wdiget.feature_selection_widget
    assert len(feature_selection_items) == len(layer.features.columns)

    # clear layers to make sure cleanup works
    viewer.layers.clear()

    assert widget.feature_selection_widget.count() == 0


def test_layer_update(make_napari_viewer, widget_config):
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

    WidgetClass = widget_config["widget_class"]
    widget = WidgetClass(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    selection_permutations = [
        [points_layer1],
        [points_layer2],
        [points_layer1, points_layer2],
    ]

    for possible_selection in selection_permutations:
        viewer.layers.selection = possible_selection

        feature_selection_items = [
            widget.feature_selection_widget.item(i).text()
            for i in range(widget.feature_selection_widget.count())
        ]

        for feature in widget.common_columns:
            # make sure common columns are in every layer
            assert feature in feature_selection_items
            for layer in possible_selection:
                assert feature in layer.features.columns

        # make sure that the name of the layer is in the collected features
        collected_features = widget._get_features()
        for layer in possible_selection:
            for feature in widget.common_columns:
                assert feature in collected_features.columns
                assert layer.unique_id in collected_features["layer"].values


def test_feature_update(make_napari_viewer, widget_config):
    viewer = make_napari_viewer()

    points_layer = create_points()
    viewer.add_layer(points_layer)

    WidgetClass = widget_config["widget_class"]
    widget = WidgetClass(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    # select all possible permutations of features
    combinations = [
        [0],
        [0, 1],
        [0, 1, 2],
    ]

    for combination in combinations:
        for idx in combination:
            # select the feature
            item = widget.feature_selection_widget.item(idx)
            item.setSelected(True)

            features = widget._update_features()
            selected_features = [
                item.text()
                for item in widget.feature_selection_widget.selectedItems()
            ]
            for feature in features.columns:
                assert feature in selected_features

            for selected_feature in selected_features:
                assert selected_feature in features.columns
                assert selected_feature in points_layer.features.columns
                assert (
                    selected_feature
                    in widget.selected_algorithm_widget.data.value.columns
                )


def test_algorithm_change(make_napari_viewer, widget_config):

    viewer = make_napari_viewer()

    points_layer = create_points()
    viewer.add_layer(points_layer)

    WidgetClass = widget_config["widget_class"]
    widget = WidgetClass(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    # select all possible permutations of features
    for idx in range(widget.algorithm_selection.count()):
        widget.algorithm_selection.setCurrentIndex(idx)
        assert (
            widget.selected_algorithm
            == widget.algorithm_selection.itemText(idx)
        )


def test_algorithm_execution(make_napari_viewer, qtbot, widget_config):
    from qtpy.QtCore import QEventLoop

    viewer = make_napari_viewer()

    layer = create_points()
    viewer.add_layer(layer)

    WidgetClass = widget_config["widget_class"]
    widget = WidgetClass(viewer)
    qtbot.addWidget(widget)

    # Select features in the QListWidget and the algorithm in the QComboBox
    widget.feature_selection_widget.selectAll()

    for algorithm in widget.algorithms:
        widget.algorithm_selection.setCurrentText(algorithm)

        # Create an event loop to wait for the process to finish
        loop = QEventLoop()

        # Connect the worker's finished signal to the loop quit slot
        def on_worker_finished(loop=loop):
            loop.quit()

        # Wait until the worker is created and then connect the finished signal
        def on_button_clicked():
            if widget.worker:
                widget.worker.finished.connect(on_worker_finished)

        # Connect the button clicked signal to the function that will
        # connect the worker's finished signal
        widget.selected_algorithm_widget.call_button.clicked.connect(
            on_button_clicked
        )
        widget.selected_algorithm_widget.call_button.clicked.emit()

        # Start the loop and wait for the worker to finish and the loop to quit
        loop.exec_()
        qtbot.wait(100)

        # Check if the results are added to the layer
        column_prefix = widget.algorithms[algorithm]["column_string"]
        for col in layer.features.columns:
            if col.startswith(column_prefix):
                break
        else:
            raise AssertionError(
                f"Results not found in layer features for algorithm {algorithm}"
            )

        # if a clustering algorithm is executed, assert that the resulting
        # columns are of type "category"
        if widget_config["widget_class"] == ClusteringWidget:
            assert layer.features[col].dtype.name == "category"

            # check that there are no -1 values in the clustering results
            assert not any(
                layer.features[col] == -1
            ), f"-1 values found in clustering results for {algorithm}"
