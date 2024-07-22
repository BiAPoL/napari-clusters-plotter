from napari.qt.threading import thread_worker, FunctionWorker
import pandas as pd

def reduce_pca(
        data: pd.DataFrame,
        n_components: int = 2,
        scale: bool = True) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using PCA
    """

    @thread_worker(progress=True)
    def _reduce_pca(data: pd.DataFrame, n_components: int, scale: bool) -> FunctionWorker[pd.DataFrame]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        if scale:
            data = StandardScaler().fit_transform(data.values)
        else:
            data = data.values

        pca = PCA(n_components=n_components)
        pca.fit(data)
        return pca.transform(data)
    
    return _reduce_pca(data, n_components, scale)

def reduce_tsne(
        data: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 30,
        scale: bool = True) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using t-SNE
    """

    @thread_worker(progress=True)
    def _reduce_tsne(data: pd.DataFrame, n_components: int, perplexity: float, scale: bool) -> pd.DataFrame:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        print('working on tsne')
        if scale:
            data = StandardScaler().fit_transform(data)
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(data)

    return _reduce_tsne(data, n_components, perplexity, scale)

def reduce_umap(
        data: pd.DataFrame,
        n_components: int = 2,
        n_neighbors: int = 30,
        scale: bool = True) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using UMAP
    """

    @thread_worker(progress=True)
    def _reduce_umap(
        data: pd.DataFrame,
        n_components: int,
        n_neighbors: int,
        scale: bool) -> pd.DataFrame:
        import umap
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
        return reducer.fit_transform(data)
    return _reduce_umap(data, n_components, n_neighbors, scale)
