DEFAULTS_CLUSTERING = {
    "kmeans_nr_clusters": 2,
    "kmeans_nr_iterations": 300,
    "standardization": False,
    "hdbscan_min_clusters_size": 5,
    "hdbscan_min_nr_samples": 5,
    "gmm_nr_clusters": 2,
    "ms_quantile": 0.2,
    "ms_n_samples": 50,
    "ac_n_clusters": 2,
    "ac_n_neighbors": 2,
    "custom_name": "",
}

DEFAULTS_DIM_REDUCTION = {
    "n_neighbors": 15,
    "perplexity": 30,
    "standardization": True,
    "pca_components": 0,
    "explained_variance": 95.0,
    "n_components": 2,
    # enabling multithreading for UMAP can result in crashing kernel if napari is opened from the Jupyter notebook,
    # therefore by default the following value is False.
    # See more: https://github.com/BiAPoL/napari-clusters-plotter/issues/169
    "umap_separate_thread": False,
    "min_distance_umap": 0.1,
    "mds_n_init": 4,
    "mds_metric": True,
    "mds_max_iter": 300,
    "mds_eps": 0.001,
    "custom_name": "",
}

ID_NAME = "_CLUSTER_ID"
_POINTER = "frame"
EXCLUDE = [ID_NAME, _POINTER, "UMAP", "t-SNE", "PCA"]

