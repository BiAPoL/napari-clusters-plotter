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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data.values)
        else:
            preprocessed = non_nan_data.values

        pca = PCA(n_components=n_components)
        pca.fit(preprocessed)
        reduced_data = pca.transform(preprocessed)

        # Add NaN rows back
        result = pd.DataFrame(index=data.index, columns=range(n_components))
        result.loc[non_nan_data.index] = reduced_data

        return result

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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        reduced_data = tsne.fit_transform(preprocessed)

        # Add NaN rows back
        result = pd.DataFrame(index=data.index, columns=range(n_components))
        result.loc[non_nan_data.index] = reduced_data

        return result

    return _reduce_tsne(data, n_components, perplexity, scale)


def reduce_umap(
    data: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    scale: bool = True,
) -> FunctionWorker[pd.DataFrame]:
    """
    Reduce the data using UMAP
    """

    @thread_worker(progress=True)
    def _reduce_umap(
        data: pd.DataFrame,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        scale: bool,
    ) -> pd.DataFrame:
        import umap
        from sklearn.preprocessing import StandardScaler

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        reduced_data = reducer.fit_transform(preprocessed)

        # Add NaN rows back
        result = pd.DataFrame(index=data.index, columns=range(n_components))
        result.loc[non_nan_data.index] = reduced_data

        return result

    return _reduce_umap(data, n_components, n_neighbors, min_dist, scale)


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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        # Perform KMeans clustering (+1 to start clusters from 1)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(preprocessed) + 1

        # Add NaN rows back
        result = pd.Series(index=data.index, dtype=int)
        result.loc[non_nan_data.index] = clusters

        return result

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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        # Perform HDBSCAN clustering (+1 to start clusters from 1)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        clusters = clusterer.fit_predict(preprocessed) + 1

        # Add NaN rows back
        result = pd.Series(index=data.index, dtype=int)
        result.loc[non_nan_data.index] = clusters

        return result

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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        # Perform Gaussian Mixture clustering (+1 to start clusters from 1)
        gmm = GaussianMixture(n_components=n_components)
        clusters = gmm.fit_predict(preprocessed) + 1

        # Add NaN rows back
        result = pd.Series(index=data.index, dtype=int)
        result.loc[non_nan_data.index] = clusters

        return result

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

        # Remove NaN rows
        non_nan_data = data.dropna()

        if scale:
            preprocessed = StandardScaler().fit_transform(non_nan_data)
        else:
            preprocessed = non_nan_data.values

        # Perform Spectral Clustering (+1 to start clusters from 1)
        clusterer = SpectralClustering(n_clusters=n_clusters)
        clusters = clusterer.fit_predict(preprocessed) + 1

        # Add NaN rows back
        result = pd.Series(index=data.index, dtype=int)
        result.loc[non_nan_data.index] = clusters

        return result

    return _cluster_spectral(data, n_clusters, scale)
