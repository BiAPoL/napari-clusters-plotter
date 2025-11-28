from pathlib import Path
from typing import List
from skimage import io
import numpy as np
import pandas as pd
import pooch
import os
import zipfile
from pathlib import Path
from napari_clusters_plotter import __version__

# parse version
if 'dev' in __version__:
    from packaging.version import parse
    major, minor, patch = parse(__version__).release
    version = f"{major}.{minor}.{patch-1}"
else:
    version = __version__

DATA_REGISTRY = pooch.create(
    path=pooch.os_cache("napari-clusters-plotter"),
    base_url=f"https://github.com/biapol/napari-clusters-plotter/releases/download/v{version}/",
    registry={"sample_data.zip": "sha256:d21889252cc439b32dacbfb2d4085057da1fe28e3c35f94fee1487804cfe9615"},
)

def load_image(fname):
    zip_path = DATA_REGISTRY.fetch("sample_data.zip")

    # check if has been unzipped before
    if not os.path.exists(zip_path.split(".zip")[0]):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(zip_path.split(".zip")[0])

    fname = os.path.join(zip_path.split(".zip")[0], fname)
    image = io.imread(fname)

    return image

def load_tabular(fname, **kwargs):
    zip_path = DATA_REGISTRY.fetch("sample_data.zip")

    # check if has been unzipped before
    if not os.path.exists(zip_path.split(".zip")[0]):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(zip_path.split(".zip")[0])

    fname = os.path.join(zip_path.split(".zip")[0], fname)
    data = pd.read_csv(fname, **kwargs)
    return data

def load_registry():
    zip_path = DATA_REGISTRY.fetch("sample_data.zip")

    # check if has been unzipped before
    if not os.path.exists(zip_path.split(".zip")[0]):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(zip_path.split(".zip")[0])

    fname = os.path.join(zip_path.split(".zip")[0], "sample_data/data_registry.txt")
    registry = pd.read_csv(fname, sep=': sha256:', header=None)
    registry.columns = ['file', 'hash']
    return registry

def skan_skeleton() -> List["LayerData"]:  # noqa: F821

    df_paths = load_tabular("shapes_skeleton/all_paths.csv")
    df_features = load_tabular("shapes_skeleton/skeleton_features.csv", index_col="Unnamed: 0")

    # skeleton_id column should be categorical
    categorical_columns = [
        "skeleton_id",
        "node_id_src",
        "branch_type",
        "path_id",
        "random_path_id",
    ]
    for feature in categorical_columns:
        df_features[feature] = df_features[feature].astype("category")

    list_of_paths = []
    shape_types = []
    for _, group in list(df_paths.groupby("index")):
        list_of_paths.append(group[["axis-0", "axis-1", "axis-2"]].values)
        shape_types.append(group["shape-type"].values[0])

    layer_paths = (
        list_of_paths,
        {
            "name": "shapes_skeleton",
            "shape_type": shape_types,
            "features": df_features,
            "edge_width": 0.25,
            "blending": "translucent",
        },
        "shapes",
    )

    layer_blobs = (
        load_image("shapes_skeleton/blobs.tif"),
        {
            "name": "binary blobs",
            "opacity": 0.5,
            "blending": "additive",
        },
        "labels",
    )

    return [layer_paths, layer_blobs]


def tgmm_mini_dataset() -> List["LayerData"]:  # noqa: F821
    
    features = load_tabular(
        "tracking_data/tgmm-mini-spot.csv",
        skiprows=[1, 2],
        low_memory=False,
        encoding="utf-8")
    data = load_tabular("tracking_data/tgmm-mini-tracks-layer-data.csv")

    categorical_columns = [
        "Label",
        "ID",
        "Branch spot ID",
        "Spot track ID",
    ]
    for feature in categorical_columns:
        features[feature] = features[feature].astype("category")
    tracking_label_image = load_image("tracking_data/tgmm-mini.tif")

    layer_data_tuple_tracks = (
        data,
        {
            "name": "tgmm-mini-tracks",
            "features": features,
            "scale": [5, 1, 1],
        },
        "tracks",
    )

    layer_data_tuple_labels = (
        tracking_label_image,
        {
            "name": "tgmm-mini-labels",
            "features": features,
            "scale": [5, 1, 1],
        },
        "labels",
    )

    return [layer_data_tuple_tracks, layer_data_tuple_labels]


def bbbc_1_dataset() -> List["LayerData"]:  # noqa: F821
    # read data registry file
    registry = load_registry()

    registry_bbby1 = registry[registry['file'].str.contains("BBBC007_v1_images")]
    tif_files = registry_bbby1[registry_bbby1['file'].str.endswith(".tif")]['file'].to_list()
    raw_images = [f for f in tif_files if "labels" not in f]

    n_rows = np.ceil(np.sqrt(len(raw_images)))
    n_cols = np.ceil(len(raw_images) / n_rows)

    layers = []

    images = [load_image(f) for f in raw_images]
    labels = [load_image(f.replace(".tif", "_labels.tif")) for f in raw_images]
    features = [
        load_tabular(f.replace(".tif", "_features.csv")) for f in raw_images
    ]

    max_size = max([image.shape[0] for image in images])

    for idx, (image, label, feature) in enumerate(
        zip(images, labels, features)
    ):

        translate_img_x = image.shape[0] / 2
        translate_img_y = image.shape[1] / 2

        # calculate translate in grid
        margin = 0.1 * image.shape[0]  # 10% margin
        i_row = idx // n_cols
        i_col = idx % n_cols
        translate_x = i_row * (max_size + margin) - translate_img_x
        translate_y = i_col * (max_size + margin) - translate_img_y

        ldtuple_image = (
            image,
            {
                "name": Path(raw_images[idx]).stem,
                "translate": (translate_x, translate_y),
            },
            "image",
        )

        ldtuple_labels = (
            label,
            {
                "name": Path(raw_images[idx]).stem + "_labels",
                "translate": (translate_x, translate_y),
                "features": feature,
            },
            "labels",
        )

        layers.append(ldtuple_image)
        layers.append(ldtuple_labels)

    return layers


def cells3d_curvatures() -> List["LayerData"]:  # noqa: F821
    vertices = load_tabular("cells3d/vertices.txt", sep=' ', header=None).to_numpy()
    faces = load_tabular("cells3d/faces.txt", sep=' ', header=None).to_numpy().astype(int)
    hks = load_tabular("cells3d/signature.csv")
    nuclei = load_image("cells3d/nucleus.tif")

    # create layer data tuples
    layer_data_surface = (
        (vertices, faces),
        {
            "name": "cells_3d_mitotic_nucleus_surface_curvatures",
            "features": hks,
        },
        "surface",
    )

    layer_data_nuclei = (
        nuclei,
        {
            "name": "cells_3d_nucleus",
            "colormap": "gray",
        },
        "image",
    )

    return [layer_data_nuclei, layer_data_surface]


def granule_compression_vectors() -> List["LayerData"]:  # noqa: F821
    import numpy as np
    from napari.utils import notifications

    features = load_tabular("compression_vectors/granular_compression_test.csv")
    features["iterations"] = features["iterations"].astype("category")
    features["returnStatus"] = features["returnStatus"].astype("category")
    features["Label"] = features["Label"].astype("category")
    features.drop(columns=["PSCC"], inplace=True)

    points_4d = features[["frame", "Zpos", "Ypos", "Xpos"]].to_numpy()
    vectors_4d = features[["frame", "Zdisp", "Ydisp", "Xdisp"]].to_numpy()
    vectors_4d = np.stack([points_4d, vectors_4d], axis=1)
    vectors_4d[:, 1, 0] = 0

    layerdata_vectors = (
        vectors_4d,
        {
            "name": "granule_compression_vectors",
            "features": features,
        },
        "vectors",
    )

    notifications.show_info(
        "Granule compression vectors dataset obtained from https://zenodo.org/records/17668709"
    )

    return [layerdata_vectors]
