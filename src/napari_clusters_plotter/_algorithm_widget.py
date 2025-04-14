import pandas as pd
from magicgui import magicgui
from napari.layers import (
    Labels,
    Points,
    Shapes,
    Surface,
    Vectors,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)


class BaseWidget(QWidget):

    input_layer_types = [
        Labels,
        Points,
        Surface,
        Vectors,
        Shapes,
    ]

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.layers = []

    def _get_features(self):
        features = pd.DataFrame()
        for layer in self.layers:
            _features = layer.features[self.common_columns].copy()

            # Add layer name as a categorical column
            _features["layer"] = layer.name
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
        self.selected_algorithm_widget = magicgui(
            self.algorithms[algorithm]["callback"], call_button="Run"
        )
        self.selected_algorithm_widget.native_parent_changed.emit(self)
        self.selected_algorithm_widget.called.connect(self._wait_for_finish)
        self.layout.addWidget(self.selected_algorithm_widget.native)

        self._update_features()

    def _on_update_layer_selection(self, layer):
        self.layers = list(self.viewer.layers.selection)

        # don't do anything if no layer is selected
        if self.n_selected_layers == 0:
            return

        # check if the selected layers are of the correct type
        selected_layer_types = [
            type(layer) for layer in self.viewer.layers.selection
        ]
        for layer_type in selected_layer_types:
            if layer_type not in self.input_layer_types:
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

    @property
    def selected_algorithm(self):
        return self.algorithm_selection.currentText()
