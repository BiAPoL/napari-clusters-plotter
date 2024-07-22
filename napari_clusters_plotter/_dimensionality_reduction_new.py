import pandas as pd
from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget,
    QComboBox,
    QVBoxLayout,
    QListWidget,
    QLabel,
    QAbstractItemView)

from ._algorithms import reduce_pca, reduce_tsne, reduce_umap, cluster_gaussian_mixture, cluster_kmeans, cluster_hdbscan, cluster_spectral


class ClusteringWidget(QWidget):
    algorithms = {
        "KMeans": {
            'callback': cluster_kmeans,
            'column_string': 'KMeans'
        },
        "HDBSCAN": {
            'callback': cluster_hdbscan,
            'column_string': 'HDBSCAN'
        },
        "Gaussian Mixture": {
            'callback': cluster_gaussian_mixture,
            'column_string': 'Gaussian Mixture'
        },
        "Spectral": {
            'callback': cluster_spectral,
            'column_string': 'Spectral'
        }
    }

    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()

        self.viewer = napari_viewer
        self.layers = []
        self._selected_clustering_widget = None
        self._clustering_worker = None

        # add label and list to put in the features to be reduced
        self._label_features = QLabel("Features to cluster:")
        self._feature_selection_widget = QListWidget()  # list to put in the features to be reduced
        self._feature_selection_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # add combobox with clustering options
        self._label_clustering = QLabel("Select clustering algorithm:")
        self._clustering_selection = QComboBox()
        self._clustering_selection.addItems(["KMeans", "HDBSCAN", "Gaussian Mixture", "Spectral"])

        # add layout and combobox
        self.layout = QVBoxLayout()

        self.layout.addWidget(self._label_features)
        self.layout.addWidget(self._feature_selection_widget)

        self.layout.addWidget(self._label_clustering)
        self.layout.addWidget(self._clustering_selection)
        self.setLayout(self.layout)

        self._on_algorithm_changed(0)
        self._on_update_layer_selection(None)
        self._setup_callblacks()

    def _setup_callblacks(self):
        self.viewer.layers.selection.events.changed.connect(self._on_update_layer_selection)
        self._clustering_selection.currentIndexChanged.connect(self._on_algorithm_changed)
        self._feature_selection_widget.itemSelectionChanged.connect(self._update_features)

    def _update_features(self):
        """
        Preprocess the features before running the clustering.

        Parameters
        ----------
        features : pd.DataFrame
            The features to preprocess.

        Returns
        -------
        pd.DataFrame
            The preprocessed features.
        """
        # fet features and filter out the ones that are not selected
        selected_columns = [item.text() for item in self._feature_selection_widget.selectedItems()]
        features = self._get_features()[selected_columns]

        if self._selected_clustering_widget is not None:
            self._selected_clustering_widget.data.value = features

        return features
    
    def _wait_for_finish(self, worker):
            
            self._clustering_worker = worker
            self._clustering_worker.start()
            self._clustering_worker.returned.connect(self._process_clustering_result)

    def _process_clustering_result(self, result):
        """
        Process the result of the clustering and update the features of the selected layers.
        """
        column_name = self.algorithms[self.selected_algorithm]['column_string']
        features_clustered = pd.DataFrame(result, columns=[column_name])
        features_clustered["layer"] = self._get_features()["layer"]

        for layer in self.layers:
            current_features = layer.features
            current_features[column_name] = features_clustered[features_clustered["layer"] == layer.name][column_name]
            layer.features = current_features

    def _on_algorithm_changed(self, index):
            
            if self._selected_clustering_widget is not None:
                self.layout.removeWidget(self._selected_clustering_widget.native)
                self._selected_clustering_widget.native.deleteLater()
                
            # update the features list
            self._selected_clustering_widget = magicgui(
                self.algorithms[self._clustering_selection.currentText()]['callback'],
                call_button="Cluster"
                )
            
            self._selected_clustering_widget.native_parent_changed.emit(self)
            self._selected_clustering_widget.called.connect(self._wait_for_finish)
            self.layout.addWidget(self._selected_clustering_widget.native)

            # update the features in the reducer widget to be ready to process
            self._update_features()

    def _on_update_layer_selection(self, layer):
        # update layer selection combobox
        self.layers = list(self.viewer.layers.selection)

        # select common columns
        features_to_add = self._get_features()[self.common_columns]

        # filter out columns that pertain to the results of the clustering
        clustering_column_strings = [clustering['column_string'] for clustering in self.algorithms.values()]
        features_to_add = features_to_add.drop(
            columns=[column for column in features_to_add.columns if any(column.startswith(s) for s in clustering_column_strings)]
        )

        self._feature_selection_widget.clear()
        self._feature_selection_widget.addItems(features_to_add.columns)

        # update the features in the reducer widget to be ready to process
        self._update_features()

    def _get_features(self) -> pd.DataFrame:
        """
        Get the features from the selected layers.

        Returns
        -------
        pd.DataFrame
            The features of all selected layers.
        """

        # concatenate the features of all selected layers
        # first put all features in a list of tables and add the layer's name as a column
        features = pd.DataFrame()
        for layer in self.layers:
            _features = layer.features[self.common_columns].copy()
            _features["layer"] = layer.name
            features = pd.concat([features, _features], axis=0)

        return features.reset_index(drop=True)
    
    @property
    def selected_algorithm(self):
        return self._clustering_selection.currentText()
    
    @property
    def common_columns(self) -> list[str]:
        """
        Columns that are in all selected layers.
        """
        # find columns that are in all selected layers
        if len(self.layers) == 0:
            return []
        common_columns = [list(layer.features.columns) for layer in self.layers]
        common_columns = list(set.intersection(*map(set, common_columns)))

        return common_columns


class DimensionalityReductionWidget(QWidget):
    algorithms = {
        "PCA": {
            'callback': reduce_pca,
            'column_string': 'PC'
        },
        "t-SNE": {
            'callback': reduce_tsne,
            'column_string': 't-SNE'
        },
        "UMAP": {
            'callback': reduce_umap,
            'column_string': 'UMAP'
        }
    }
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        
        self.viewer = napari_viewer
        self.layers = []
        self._selected_reducer_widget = None
        self._reduction_worker = None

        # add label and list to put in the features to be reduced
        self._label_features = QLabel("Features to reduce:")
        self._feature_selection_widget = QListWidget()  # list to put in the features to be reduced
        self._feature_selection_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # add combobox with clustering options
        self._label_reducer = QLabel("Select dimensionality reduction algorithm:")
        self._reducer_selection = QComboBox()
        self._reducer_selection.addItems(["PCA", "t-SNE", "UMAP"])

        # add layout and combobox
        self.layout = QVBoxLayout()

        self.layout.addWidget(self._label_features)
        self.layout.addWidget(self._feature_selection_widget)

        self.layout.addWidget(self._label_reducer)
        self.layout.addWidget(self._reducer_selection)
        self.setLayout(self.layout)

        self._on_algorithm_changed(0)
        self._on_update_layer_selection(None)
        self._setup_callblacks()

    def _setup_callblacks(self):
        self.viewer.layers.selection.events.changed.connect(self._on_update_layer_selection)
        self._reducer_selection.currentIndexChanged.connect(self._on_algorithm_changed)
        self._feature_selection_widget.itemSelectionChanged.connect(self._update_features)

    def _update_features(self):
        """
        Preprocess the features before running the dimensionality reduction.

        Parameters
        ----------
        features : pd.DataFrame
            The features to preprocess.

        Returns
        -------
        pd.DataFrame
            The preprocessed features.
        """
        # fet features and filter out the ones that are not selected
        selected_columns = [item.text() for item in self._feature_selection_widget.selectedItems()]
        features = self._get_features()[selected_columns]

        if self._selected_reducer_widget is not None:
            self._selected_reducer_widget.data.value = features

        return features
    
    def _wait_for_finish(self, worker):

        self._reduction_worker = worker
        self._reduction_worker.start()
        self._reduction_worker.returned.connect(self._process_reduction_result)

    def _process_reduction_result(self, result):
        """
        Process the result of the dimensionality reduction and update the features of the selected layers.
        """
        column_names = [f"{self.algorithms[self.selected_algorithm]['column_string']}{i}" for i in range(result.shape[1])]
        features_reduced = pd.DataFrame(result, columns=column_names)
        features_reduced["layer"] = self._get_features()["layer"]

        for layer in self.layers:
            current_features = layer.features
            for column in column_names:
                current_features[column] = features_reduced[features_reduced["layer"] == layer.name][column]
            
            layer.features = current_features

    def _on_algorithm_changed(self, index):

        if self._selected_reducer_widget is not None:
            self.layout.removeWidget(self._selected_reducer_widget.native)
            self._selected_reducer_widget.native.deleteLater()
            
        # update the features list
        self._selected_reducer_widget = magicgui(
            self.algorithms[self._reducer_selection.currentText()]['callback'],
            call_button="Reduce"
            )
        self._selected_reducer_widget.native_parent_changed.emit(self)
        self._selected_reducer_widget.called.connect(self._wait_for_finish)
        
        # add reducer widget to layout at second-last position
        self.layout.addWidget(self._selected_reducer_widget.native)

        # update the features in the reducer widget to be ready to process
        self._update_features()

    def _on_update_layer_selection(self, layer):
        # update layer selection combobox
        self.layers = list(self.viewer.layers.selection)

        # select common columns
        features_to_add = self._get_features()[self.common_columns]

        # filter out columns that pertain to the results of the reduction
        reducer_column_strings = [reducer['column_string'] for reducer in self.algorithms.values()]
        features_to_add = features_to_add.drop(
            columns=[column for column in features_to_add.columns if any(column.startswith(s) for s in reducer_column_strings)]
        )

        self._feature_selection_widget.clear()
        self._feature_selection_widget.addItems(features_to_add.columns)

        # update the features in the reducer widget to be ready to process
        self._update_features()

    def _get_features(self) -> pd.DataFrame:
        """
        Get the features from the selected layers.

        Returns
        -------
        pd.DataFrame
            The features of all selected layers.
        """

        # concatenate the features of all selected layers
        # first put all features in a list of tables and add the layer's name as a column
        features = pd.DataFrame()
        for layer in self.layers:
            _features = layer.features[self.common_columns].copy()
            _features["layer"] = layer.name
            features = pd.concat([features, _features], axis=0)

        return features.reset_index(drop=True)

    @property
    def selected_algorithm(self):
        return self._reducer_selection.currentText()

    @property
    def common_columns(self) -> list[str]:
        """
        Columns that are in all selected layers.
        """
        # find columns that are in all selected layers
        if len(self.layers) == 0:
            return []
        common_columns = [list(layer.features.columns) for layer in self.layers]
        common_columns = list(set.intersection(*map(set, common_columns)))

        return common_columns
