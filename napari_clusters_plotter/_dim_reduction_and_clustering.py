import pandas as pd
from ._algorithm_widget import AlgorithmWidgetBase
from ._algorithms import reduce_pca, reduce_tsne, reduce_umap, cluster_gaussian_mixture, cluster_kmeans, cluster_hdbscan, cluster_spectral


class ClusteringWidget(AlgorithmWidgetBase):
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
        super().__init__(
            napari_viewer,
            ClusteringWidget.algorithms,
            "Features to cluster:",
            ["KMeans", "HDBSCAN", "Gaussian Mixture", "Spectral"]
        )

    def _process_result(self, result):
        """
        Process the result of the clustering algorithm and update the layer
        """
        
        column_name = self.algorithms[self.selected_algorithm]['column_string']
        features_clustered = pd.DataFrame(result, columns=[column_name])
        features_clustered["layer"] = self._get_features()["layer"]

        for layer in self.layers:
            current_features = layer.features

            # add the columns to the features
            layer_feature_subset = features_clustered[features_clustered["layer"] == layer.name]
            current_features[column_name] = layer_feature_subset[column_name].values

            # overwrite the features to trigger the features changed signal
            layer.features = current_features

class DimensionalityReductionWidget(AlgorithmWidgetBase):
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
        super().__init__(
            napari_viewer,
            DimensionalityReductionWidget.algorithms,
            "Features to reduce:",
            ["PCA", "t-SNE", "UMAP"]
        )

    def _process_result(self, result):
        """
        Process the result of the dimensionality reduction algorithm and update the layer
        """
        column_names = [f"{self.algorithms[self.selected_algorithm]['column_string']}{i}" for i in range(result.shape[1])]
        features_reduced = pd.DataFrame(result, columns=column_names)
        features_reduced["layer"] = self._get_features()["layer"].values

        for layer in self.layers:
            current_features = layer.features
            for column in column_names:
                layer_feature_subset = features_reduced[features_reduced["layer"] == layer.name]
                
                # add the columns to the features
                current_features[column] = layer_feature_subset[column].values

            # overwrite the features to trigger the features changed signal
            layer.features = current_features