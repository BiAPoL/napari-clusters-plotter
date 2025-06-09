# Contributing a new algorithm

If you want to make a new clustering or dimensionality reduction algorithm available in the napari-clusters-plotter, please follow a few guidelines and specifications. First, fork the repository and create yourself a different branch to work on. AOnce this is set up, you can find all implemented allgorithms under `src/napari_clusters_plotter/algorithms.py`, where you can add your algorithm, too.

## New dimensionality reduction algorithm

If you want to add your algorithm there, please make sure that it adheres to the following syntax:

```python
def reduce_my_algorithm(
    data: pd.DataFrame,
    your_int_algorithm_parameter: int = 2,
    your_float_algorithm_parameter: float = 0.1,
    scale: bool = True
) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using my algorithm
    """

    @thread_worker(progress=True)
    def _reduce_my_algorithm(
        data: pd.DataFrame,
        your_int_algorithm_parameter: int,
        your_float_algorithm_parameter: float,
        scale: bool
    ) -> FunctionWorker[pd.DataFrame]:
        import your_module
        from sklearn.preprocessing import StandardScaler

        # Keep this code
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data.values)
        else:
            preprocessed = non_nan_data.values

        # <<<Implement your algorithm here
        ...
        reduced_data = your_module.fit_transform(preprocessed)
        # <<<<

        # Add NaN rows back - keep this part
        result = pd.DataFrame(index=data.index, columns=range(n_components))
        result.loc[non_nan_data.index] = reduced_data

        return result

    return _reduce_my_algorithm(
        data,
        your_int_algorithm_parameter,
        your_float_algorithm_parameter,
        scale)

```

Here's a breakdown of what each part of this code does. The outer function `reduce_my_algorithm` is what will later be visible to the napari clusters plotter. The inner function (`_reduce_my_algorithm`) will be submitted to a [napari threadworker](https://napari.org/stable/guides/threading.html), which allows for the algorithm to execute in a non-blocking fashion.

Once that is done, you need to make the method available for the clustering or dimensionality reduction widget, respectively. If your algorithm is a dimensionality reduction algorithm, you'll find the relevant widget under `src/napari_clusters_plotter/_dim_reduction_and_clustering.py`. You can add your algorithm to the Dimensionality reduction widget there as follows:

```python
class DimensionalityReductionWidget(AlgorithmWidgetBase):
    algorithms = {
        "PCA": {
            "callback": reduce_pca,
            "column_string": "PC",
            "doc_url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
        },
        "t-SNE": {
            "callback": reduce_tsne,
            "column_string": "t-SNE",
            "doc_url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        },
        "UMAP": {
            "callback": reduce_umap,
            "column_string": "UMAP",
            "doc_url": "https://umap-learn.readthedocs.io/en/latest/",
        },
        "my-new-algorithm": {
            "callback": reduce_my_algorithm,
            "column_string": "acronym_of_my_algorithm",
            "doc_url": "https://link-to-my-algorithm-that-explains-what-it-does.com"
        }
    }

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer,
            DimensionalityReductionWidget.algorithms,
            "Features to reduce:",
            ["PCA", "t-SNE", "UMAP", "some_acronym_for_your_algorithm"],
        )

```

The relevant parts here are to add the above-implemented function as a callback to the widget. Secondly, you'll need to provide an acronym for your algorithm. The reduced features will then appear in the list of features as `ACRONYM_0` and `ACRONYM_1` (e.g., `PC_0` and `PC_1` for PCA). Lastly, please add a link to a documentation page that describes how your algorithm works, what it does and what its parameters mean.

## New clustering algorithm

To add a new clustering algorithm, follow the steps above analogeously for the clustering widget. For the implementation of the algorithm itself, nothing changes.

```{hint}
Clustering algorithms are expected to return a single integer column!
```

```{hint}
In the napari-clusters-plotter convention, cluster ids **start with 1** - the value 0 is reserved for unclustered data points. This means that if your algorithm returns a cluster id of 0, you should change it to 1 before returning the result.
```

This being said, a clustering algorithm should look like this:

```python
def cluster_method(
    data: pd.DataFrame, n_clusters: int = 3, scale: bool = True
) -> FunctionWorker[pd.Series]:
    """
    Cluster the data using Spectral Clustering
    """

    @thread_worker(progress=True)
    def _cluster_method(
        data: pd.DataFrame, some_parameter: int, scale: bool
    ) -> pd.Series:
        from module import MyClusteringAlgorithm

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        # Perform Spectral Clustering (+1 to start clusters from 1)
        clusterer = MyClusteringAlgorithm(some_parameter=some_parameter)
        clusters = clusterer.fit_predict(preprocessed) + 1

        # Add NaN rows back
        result = pd.Series(index=data.index, dtype=int)
        result.loc[non_nan_data.index] = clusters

        return result

    return _cluster_method(data, n_clusters, scale)
```
