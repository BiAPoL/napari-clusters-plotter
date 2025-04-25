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
        "PCA": {"callback": reduce_pca, "column_string": "PC"},
        "t-SNE": {"callback": reduce_tsne, "column_string": "t-SNE"},
        "UMAP": {"callback": reduce_umap, "column_string": "UMAP"},
        "My algorithm" : {"callback": reduce_my_algorithm, "column_string": "some_acronym_for_your_algorithm"}
    }

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer,
            DimensionalityReductionWidget.algorithms,
            "Features to reduce:",
            ["PCA", "t-SNE", "UMAP", "some_acronym_for_your_algorithm"],
        )

```

The relevant parts here are to add the above-implemented function as a callback to the widget. Secondly, you'll need to provide an acronym for your algorithm. The reduced features will then appear in the list of features as `ACRONYM_0` and `ACRONYM_1` (e.g., `PC_0` and `PC_1` for PCA).

## New clustering algorithm

To add a new clustering algorithm, follow the steps above analogeously for the clustering widget. For the implementation of the algorithm itself, nothing changes.

**Note**: Clustering algorithms are expected to return a single integer column!
