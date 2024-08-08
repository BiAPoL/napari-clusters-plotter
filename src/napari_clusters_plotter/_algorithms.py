import pandas as pd
from napari.qt.threading import FunctionWorker, thread_worker


def reduce_pca(
    data: pd.DataFrame, n_components: int = 2, scale: bool = True
) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using PCA
    """

    @thread_worker(progress=True)
    def _reduce_pca(
        data: pd.DataFrame, n_components: int, scale: bool
    ) -> FunctionWorker[pd.DataFrame]:
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
    scale: bool = True,
) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using t-SNE
    """

    @thread_worker(progress=True)
    def _reduce_tsne(
        data: pd.DataFrame, n_components: int, perplexity: float, scale: bool
    ) -> pd.DataFrame:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        print("working on tsne")
        if scale:
            data = StandardScaler().fit_transform(data)
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(data)

    return _reduce_tsne(data, n_components, perplexity, scale)


def reduce_umap(
    data: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 30,
    scale: bool = True,
) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using UMAP
    """

    @thread_worker(progress=True)
    def _reduce_umap(
        data: pd.DataFrame, n_components: int, n_neighbors: int, scale: bool
    ) -> pd.DataFrame:
        import umap
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
        return reducer.fit_transform(data)

    return _reduce_umap(data, n_components, n_neighbors, scale)


def cluster_kmeans(
    data: pd.DataFrame, n_clusters: int = 3, scale: bool = True
) -> FunctionWorker[pd.Series]:
    """
    Cluster the data using KMeans
    """

    @thread_worker(progress=True)
    def _cluster_kmeans(
        data: pd.DataFrame, n_clusters: int, scale: bool
    ) -> pd.Series:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(data)

    return _cluster_kmeans(data, n_clusters, scale)


def cluster_hdbscan(
    data: pd.DataFrame,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    scale: bool = True,
) -> FunctionWorker[pd.Series]:
    """
    Cluster the data using HDBSCAN
    """

    @thread_worker(progress=True)
    def _cluster_hdbscan(
        data: pd.DataFrame,
        min_cluster_size: int,
        min_samples: int,
        scale: bool,
    ) -> pd.Series:
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        return clusterer.fit_predict(data)

    return _cluster_hdbscan(data, min_cluster_size, min_samples, scale)


def cluster_gaussian_mixture(
    data: pd.DataFrame, n_components: int = 3, scale: bool = True
) -> FunctionWorker[pd.Series]:
    """
    Cluster the data using Gaussian Mixture
    """

    @thread_worker(progress=True)
    def _cluster_gaussian_mixture(
        data: pd.DataFrame, n_components: int, scale: bool
    ) -> pd.Series:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        gmm = GaussianMixture(n_components=n_components)
        return gmm.fit_predict(data)

    return _cluster_gaussian_mixture(data, n_components, scale)


def cluster_spectral(
    data: pd.DataFrame, n_clusters: int = 3, scale: bool = True
) -> FunctionWorker[pd.Series]:
    """
    Cluster the data using Spectral Clustering
    """

    @thread_worker(progress=True)
    def _cluster_spectral(
        data: pd.DataFrame, n_clusters: int, scale: bool
    ) -> pd.Series:
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler

        if scale:
            data = StandardScaler().fit_transform(data)

        clusterer = SpectralClustering(n_clusters=n_clusters)
        return clusterer.fit_predict(data)

    return _cluster_spectral(data, n_clusters, scale)
