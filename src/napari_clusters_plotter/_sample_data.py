import glob
import os
from pathlib import Path
from typing import List


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
    tracking_label_image = imread(path / Path("tgmm-mini.tif"))

    layer_data_tuple_tracks = (
        data,
        {
            "name": "tgmm-mini-tracks",
            "features": features,
        },
        "tracks",
    )

    layer_data_tuple_labels = (
        tracking_label_image,
        {
            "name": "tgmm-mini-labels",
            "features": features,
        },
        "labels",
    )

    return [layer_data_tuple_tracks, layer_data_tuple_labels]


def bbbc_1_dataset() -> List["LayerData"]:  # noqa: F821
    import numpy as np
    import pandas as pd
    from skimage import io

    # get path of this file
    path = Path(__file__).parent / "sample_data" / "BBBC007_v1_images"

    tif_files = glob.glob(
        os.path.join(str(path), "**", "*.tif"), recursive=True
    )
    raw_images = [f for f in tif_files if "labels" not in f]

    spacing = 500
    # calculate positions of images on grid
    n_cols = np.sqrt(len(raw_images))
    image_layers = []
    labels_layers = []

    for i, raw_image_filename in enumerate(raw_images):

        label_filename = raw_image_filename.replace(".tif", "_labels.tif")
        feature_filename = raw_image_filename.replace(".tif", "_features.csv")
        image = io.imread(raw_image_filename)
        labels = io.imread(label_filename)

        row = i // n_cols
        col = i % n_cols

        features = pd.read_csv(feature_filename)

        ldtuple_image = (
            image,
            {
                "name": Path(raw_image_filename).stem,
                "translate": [spacing * col, spacing * row],
            },
            "image",
        )

        ldtuple_labels = (
            labels,
            {
                "name": Path(raw_image_filename).stem + "_labels",
                "translate": [spacing * col, spacing * row],
                "features": features,
            },
            "labels",
        )

        image_layers.append(ldtuple_image)
        labels_layers.append(ldtuple_labels)

    return image_layers + labels_layers


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
    layer_data_surface = [
        (vertices, faces),
        {
            "name": "cells_3d_mitotic_nucleus_surface_curvatures",
            'features': hks,},
         "surface",
    ]

    layer_data_nuclei = (
        nuclei,
        {
            "name": "cells_3d_nucleus",
            "colormap": "gray",
        },
        "image",
    )

    return [layer_data_nuclei, layer_data_surface]
