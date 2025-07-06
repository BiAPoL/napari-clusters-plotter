import glob
import os
from pathlib import Path
from typing import List

import numpy as np


def skan_skeleton() -> List["LayerData"]:  # noqa: F821
    import pandas as pd
    from skimage.io import imread

    paths_data = Path(__file__).parent / "sample_data" / "shapes_skeleton"
    df_paths = pd.read_csv(
        paths_data / Path("all_paths.csv"),
    )
    df_features = pd.read_csv(
        paths_data / Path("skeleton_features.csv"),
        index_col="Unnamed: 0",  # Adjusted to match the CSV structure
    )

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
        imread(paths_data / Path("blobs.tif")),
        {
            "name": "binary blobs",
            "opacity": 0.5,
            "blending": "additive",
        },
        "labels",
    )

    return [layer_paths, layer_blobs]


def tgmm_mini_dataset() -> List["LayerData"]:  # noqa: F821
    import pandas as pd
    from skimage.io import imread

    path = Path(__file__).parent / "sample_data" / "tracking_data"
    data = pd.read_csv(path / Path("tgmm-mini-tracks-layer-data.csv"))
    features = pd.read_csv(
        path / Path("tgmm-mini-spot.csv"),
        skiprows=[1, 2],
        low_memory=False,
        encoding="utf-8",
    )

    categorical_columns = [
        "Label",
        "ID",
        "Branch spot ID",
        "Spot track ID",
    ]
    for feature in categorical_columns:
        features[feature] = features[feature].astype("category")
    tracking_label_image = imread(path / Path("tgmm-mini.tif"))

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
    import pandas as pd
    from skimage import io

    # get path of this file
    path = Path(__file__).parent / "sample_data" / "BBBC007_v1_images"

    tif_files = glob.glob(
        os.path.join(str(path), "**", "*.tif"), recursive=True
    )
    raw_images = [f for f in tif_files if "labels" not in f]

    n_rows = np.ceil(np.sqrt(len(raw_images)))
    n_cols = np.ceil(len(raw_images) / n_rows)

    layers = []

    images = [io.imread(f) for f in raw_images]
    labels = [io.imread(f.replace(".tif", "_labels.tif")) for f in raw_images]
    features = [
        pd.read_csv(f.replace(".tif", "_features.csv")) for f in raw_images
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
    import numpy as np
    import pandas as pd
    from skimage import io

    path = Path(__file__).parent / "sample_data" / "cells3d"

    # load data
    vertices = np.loadtxt(path / "vertices.txt")
    faces = np.loadtxt(path / "faces.txt").astype(int)
    hks = pd.read_csv(path / "signature.csv")
    nuclei = io.imread(path / "nucleus.tif")

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
