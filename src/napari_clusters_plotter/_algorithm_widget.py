import warnings

import pandas as pd
from magicgui import magic_factory
from magicgui.widgets import Label
from napari.layers import (
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)


class BaseWidget(QWidget):

    input_layer_types = [Labels, Points, Surface, Vectors, Shapes, Tracks]

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.layers = []

    def _get_features(self):
        features = pd.DataFrame()
        for layer in self.layers:
            _features = layer.features[self.common_columns].copy()

            # Add layer name as a categorical column
            _features["layer"] = layer.unique_id
            _features["layer"] = _features["layer"].astype("category")
            features = pd.concat([features, _features], axis=0)

        # make sure that MANUAL_CLUSTER_ID is always categorical
        if "MANUAL_CLUSTER_ID" in features.columns:
            features["MANUAL_CLUSTER_ID"] = features[
                "MANUAL_CLUSTER_ID"
            ].astype("category")
        return features.reset_index(drop=True)

    def _clean_up(self):
        """Determines what happens in case of no layer selected"""

        raise NotImplementedError(
            "This function should be implemented in the subclass."
        )

    @property
    def common_columns(self):
        if len(self.layers) == 0:
            return []
        common_columns = [
            list(layer.features.columns) for layer in self.layers
        ]
        common_columns = list(set.intersection(*map(set, common_columns)))
        return common_columns

    @property
    def categorical_columns(self):
        if len(self.layers) == 0:
            return []
        return self._get_features().select_dtypes(include="category").columns

    @property
    def n_selected_layers(self) -> int:
        """
        Number of currently selected layers.
        """
        return len(list(self.viewer.layers.selection))

    def get_valid_layers(self):
        """
        Check if the currently selected layers are of the correct type.
        """
        return [
            layer
            for layer in self.viewer.layers.selection
            if self._is_supported_layer(layer)
        ]

    def _is_supported_layer(self, layer: Layer) -> bool:
        """
        Check if the layer is of a supported type. Supported types are
        Labels, Points, Shapes, Surface, Tracks, and Vectors as well as
        any custom layer that inherits from these types.
        """
        return any(
            isinstance(layer, layer_type)
            for layer_type in self.input_layer_types
        )


class AlgorithmWidgetBase(BaseWidget):
    def __init__(self, napari_viewer, algorithms, label_text, combo_box_items):
        super().__init__(napari_viewer)

        self.selected_algorithm_widget = None
        self.worker = None

        # Add label and list to put in the features to be reduced
        self.label_features = QLabel(label_text)
        self.feature_selection_widget = QListWidget()
        self.feature_selection_widget.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        # Add combobox with algorithm options
        self.label_algorithm = QLabel(
            f"Select {label_text.split(' ')[-2]} algorithm:"
        )
        self.algorithm_selection = QComboBox()
        self.algorithm_selection.addItems(combo_box_items)

        # Add layout and combobox
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label_features)
        self.layout.addWidget(self.feature_selection_widget)
        self.layout.addWidget(self.label_algorithm)
        self.layout.addWidget(self.algorithm_selection)
        self.setLayout(self.layout)

        self.algorithms = algorithms

        self._on_algorithm_changed(0)
        self._on_update_layer_selection(None)
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.viewer.layers.selection.events.changed.connect(
            self._on_update_layer_selection
        )
        self.algorithm_selection.currentIndexChanged.connect(
            self._on_algorithm_changed
        )
        self.feature_selection_widget.itemSelectionChanged.connect(
            self._update_features
        )

    def _update_features(self):
        """
        Update the features to be used in the selected algorithm. Called when
        the user selects a different set of features.
        """
        selected_columns = [
            item.text()
            for item in self.feature_selection_widget.selectedItems()
        ]
        features = self._get_features()[selected_columns]

        if self.selected_algorithm_widget is not None:
            self.selected_algorithm_widget.data.value = features

        return features

    def _wait_for_finish(self, worker):
        # escape empty input data
        if self.selected_algorithm_widget.data.value.empty:
            warnings.warn(
                "No features selected. Please select features before running the algorithm.",
                stacklevel=1,
            )
            return
        self.worker = worker
        self.worker.start()
        self.worker.returned.connect(self._process_result)

    def _process_result(self, result):
        raise NotImplementedError("Subclasses should implement this method.")

    def _on_algorithm_changed(self, index):
        if self.selected_algorithm_widget is not None:
            self.layout.removeWidget(self.selected_algorithm_widget.native)
            self.selected_algorithm_widget.native.deleteLater()

        algorithm = self.algorithm_selection.currentText()
        widget_factory = magic_factory(
            self.algorithms[algorithm]["callback"],
            call_button="Run",
            widget_init=lambda widget: self._on_init_algorithm(widget),
        )
        self.selected_algorithm_widget = widget_factory()
        self.selected_algorithm_widget.native_parent_changed.emit(self)
        self.selected_algorithm_widget.called.connect(self._wait_for_finish)
        self.layout.addWidget(self.selected_algorithm_widget.native)

        self._update_features()

    def _on_init_algorithm(self, widget):
        """
        Add a label with the documentation link to the algorithm widget.

        Taken from https://github.com/guiwitz/napari-skimage/blob/main/src/napari_skimage/skimage_detection_widget.py

        Parameters
        ----------
        widget : magicgui.widgets.Widget
            The widget to add the label to.
        """
        label_widget = Label(value="")

        algorithm = self.algorithms[self.algorithm_selection.currentText()]

        label_widget.value = (
            f'Doc pages: <a href="{algorithm["doc_url"]}" '
            f'style="color: white;">{algorithm["doc_url"]}</a>'
        )
        label_widget.native.setTextFormat(Qt.RichText)
        label_widget.native.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label_widget.native.setOpenExternalLinks(True)
        widget.extend([label_widget])

    def _on_update_layer_selection(self, layer):
        self.layers = self.get_valid_layers()
        if len(self.layers) == 0:
            self._clean_up()
            return

        # don't do anything if no layer is selected
        if self.n_selected_layers == 0:
            self._clean_up()
            return

        features_to_add = self._get_features()[self.common_columns]
        column_strings = [
            algo["column_string"] for algo in self.algorithms.values()
        ]
        features_to_add = features_to_add.drop(
            columns=[
                col
                for col in features_to_add.columns
                if any(col.startswith(s) for s in column_strings)
            ]
        )
        self.feature_selection_widget.clear()
        self.feature_selection_widget.addItems(sorted(features_to_add.columns))
        self._update_features()

    def _clean_up(self):
        """
        Clean up the widget when it is closed.
        """

        # block signals for feature selection
        self.feature_selection_widget.blockSignals(True)
        self.feature_selection_widget.clear()
        self.feature_selection_widget.blockSignals(False)

    @property
    def selected_algorithm(self):
        return self.algorithm_selection.currentText()
