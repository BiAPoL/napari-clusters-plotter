import warnings
from functools import wraps
from typing import Tuple

import napari.layers
import numpy as np
import pandas as pd
from napari_skimage_regionprops import relabel
from qtpy.QtWidgets import QListWidgetItem


def buttons_active(*buttons, active):
    for button in buttons:
        try:
            if button.isVisible():
                button.setEnabled(active)
        except RuntimeError:
            # necessary for tests because buttons are deleted before computation is finished in the secondary thread
            break


def widgets_active(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


def widgets_valid(*widgets, valid):
    for widget in widgets:
        widget.native.setStyleSheet("" if valid else "background-color: lightcoral")


def show_table(viewer, labels_layer):
    from napari_skimage_regionprops import add_table

    add_table(labels_layer, viewer)


def restore_defaults(widget, defaults: dict):
    for item, val in defaults.items():
        getattr(widget, item).value = val
        if item == "custom_name":
            widget.custom_name.clear()


def set_features(layer, tabular_data):
    if hasattr(layer, "properties"):
        layer.properties = tabular_data
    if hasattr(layer, "features"):
        layer.features = tabular_data


def get_layer_tabular_data(layer):
    if hasattr(layer, "properties") and layer.properties is not None:
        return pd.DataFrame(layer.properties)
    if hasattr(layer, "features") and layer.features is not None:
        return layer.features
    return None


def add_column_to_layer_tabular_data(layer, column_name, data):
    if hasattr(layer, "properties"):
        layer.properties[column_name] = data
    if hasattr(layer, "features"):
        layer.features.loc[:, column_name] = data


def catch_NaNs(func):
    "Remove NaNs from array for processing and put result to correct location."

    @wraps(func)
    def wrapper(*args, **kwargs):
        measurements = args[0].copy()  # this should be a DataFrame

        if isinstance(measurements, np.ndarray):
            measurements = pd.DataFrame(measurements)
        non_nan_entries = measurements.dropna().index

        new_args = list(args)
        new_args[0] = measurements.dropna()
        embedded = func(*new_args, **kwargs)

        result = pd.DataFrame(embedded[1], index=non_nan_entries)
        result = result.reindex(np.arange(len(measurements)))

        return embedded[0], result.to_numpy().squeeze()

    return wrapper


# TODO docstring
def update_properties_list(widget, exclude_list):
    selected_layer = widget.labels_select.value

    if selected_layer is not None:
        features = get_layer_tabular_data(selected_layer)
        if features is not None:
            old_selected_props = [
                i.text() for i in widget.properties_list.selectedItems()
            ]
            widget.properties_list.clear()
            for p in list(features.keys()):
                exclude = False
                for flag in exclude_list + ["index", "label"]:
                    if flag in p:
                        exclude = True
                        break
                if exclude:
                    continue
                item = QListWidgetItem(p)
                widget.properties_list.addItem(item)
                if not old_selected_props:
                    item.setSelected(True)
                    continue
                if p in old_selected_props:
                    item.setSelected(True)


def generate_cluster_image_from_layer(layer, predictionlist) -> Tuple[np.ndarray, str]:
    """
    Returns a label image where each label value corresponds
    to the cluster identity defined by the predictionlist.
    it is assumed that len(predictionlist) == max(label_image)

    Parameters
    ----------
    layer: napari.types.Layer
        Layer used for cluster predictions
    predictionlist: array
        Array containing cluster identities for each label
    """

    output = None
    layer_type = None
    if isinstance(layer, napari.layers.Labels):
        label_image = layer.data
        # reforming the prediction list this is done to account
        # for cluster labels that start at 0, conveniently hdbscan
        # labelling starts at -1 for noise, removing these from
        # the labels
        predictionlist_new = np.array(predictionlist) + 1
        output = relabel(label_image, list(predictionlist_new)).astype("uint32")
        layer_type = "Labels"
    elif isinstance(layer, napari.layers.Surface):
        surface_data = layer.data
        output = (surface_data[0], surface_data[1], np.array(predictionlist) + 1)
        layer_type = "Surface"
    else:
        warnings.warn("Image type not supported to view clusters as layer")

    return output, layer_type


def generate_cluster_image_from_image(label_image, predictionlist):
    """
    Returns a label image where each label value corresponds
    to the cluster identity defined by the predictionlist.
    Parameters
    ----------
    label_image: dask array
        Label image used for cluster predictions
    predictionlist: array
        Array containing cluster identities for each label
    """

    predictionlist_new = np.array(predictionlist) + 1
    return relabel(label_image, list(predictionlist_new)).astype("uint32")


def dask_cluster_image_timelapse(label_image, prediction_list_list):
    """
    Generates cluster images with delay using dask.

    Parameters
    ----------
    label_image: dask array
        Dask array of label image used for cluster predictions
    prediction_list_list: np.ndarray
        Array containing cluster identities for each label
    """
    import dask.array as da
    from dask import delayed

    sample = label_image[0]

    lazy_cluster_image = delayed(generate_cluster_image_from_image)  # lazy processor
    lazy_arrays = [
        lazy_cluster_image(frame, preds)
        for frame, preds in zip(label_image, prediction_list_list)
    ]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)

    return stack


def reshape_2D_timelapse(timelapse_2d):
    """
    Given a 2D timelapse of shape (t,y,x) returns a modified
    array of shape (t,z=1,y,x)
    """
    return timelapse_2d[:, np.newaxis, :, :]


def get_nice_colormap():
    colours_w_old_colors = [
        "#ff7f0e",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#ccebc5",
        "#ffed6f",
        "#0054b6",
        "#6aa866",
        "#ffbfff",
        "#8d472a",
        "#417239",
        "#d48fd0",
        "#8b7e32",
        "#7989dc",
        "#f1d200",
        "#a1e9f6",
        "#924c28",
        "#dc797e",
        "#b86e85",
        "#79ea30",
        "#4723b9",
        "#3de658",
        "#de3ce7",
        "#86e851",
        "#9734d7",
        "#d0f23c",
        "#3c4ce7",
        "#93d229",
        "#8551e9",
        "#eeea3c",
        "#ca56ee",
        "#2af385",
        "#ea48cd",
        "#7af781",
        "#7026a8",
        "#51d967",
        "#ad3bc2",
        "#4ab735",
        "#3b1784",
        "#afc626",
        "#3d44bc",
        "#d5cc31",
        "#6065e6",
        "#8fca40",
        "#9e2399",
        "#27ca6f",
        "#e530a4",
        "#54f2ad",
        "#c236aa",
        "#a1e76b",
        "#a96fe6",
        "#64a725",
        "#d26de1",
        "#52b958",
        "#867af4",
        "#ecbe2b",
        "#4f83f7",
        "#bbd14f",
        "#2f65d0",
        "#ddf47c",
        "#27165e",
        "#92e986",
        "#8544ad",
        "#91a824",
        "#2e8bf3",
        "#ec6e1b",
        "#2b6abe",
        "#eb3e22",
        "#43e8cf",
        "#e52740",
        "#5ef3e7",
        "#ed2561",
        "#6ceac0",
        "#681570",
        "#8eec9c",
        "#8f2071",
        "#add465",
        "#3a4093",
        "#e3ce58",
        "#5a3281",
        "#82bf5d",
        "#e1418b",
        "#3d8e2a",
        "#e86ec2",
        "#66ca7d",
        "#ae1e63",
        "#4abb81",
        "#dc3b6c",
        "#409e59",
        "#b34b9d",
        "#87a943",
        "#958df3",
        "#e59027",
        "#667edb",
        "#ddad3c",
        "#545daf",
        "#e4e68b",
        "#22123e",
        "#b9e997",
        "#6c2c76",
        "#b0c163",
        "#866ecb",
        "#5f892d",
        "#d889e2",
        "#276222",
        "#ab98ed",
        "#79801a",
        "#8f5baa",
        "#ab972e",
        "#7899e9",
        "#dc5622",
        "#4a9de3",
        "#bd2e10",
        "#54d5d6",
        "#bc2f25",
        "#40bd9c",
        "#c72e45",
        "#9ae5b4",
        "#891954",
        "#d6ecb1",
        "#0e0d2c",
        "#e9c779",
        "#193163",
        "#f07641",
        "#4ab5dc",
        "#e35342",
        "#6dd3e7",
        "#92230d",
        "#a3e9e2",
        "#951a28",
        "#48a7b4",
        "#a8421a",
        "#88c4e9",
        "#c55a2b",
        "#2e5c9d",
        "#bb8524",
        "#737bc6",
        "#c2bc64",
        "#661952",
        "#92bc82",
        "#46123b",
        "#d6e5c8",
        "#190b1f",
        "#e5a860",
        "#1d1d3c",
        "#f27c58",
        "#06121f",
        "#ebcfa3",
        "#06121f",
        "#f3a27d",
        "#06121f",
        "#eb6065",
        "#297a53",
        "#af437c",
        "#365412",
        "#be9ee2",
        "#636b24",
        "#e9a1d5",
        "#1c2c0c",
        "#e3bce6",
        "#06121f",
        "#cf8042",
        "#06121f",
        "#bfdee0",
        "#751718",
        "#80c1ab",
        "#bb3f44",
        "#2b9083",
        "#781731",
        "#618d58",
        "#93457c",
        "#7f954c",
        "#4b2a5c",
        "#c3bd83",
        "#290d1b",
        "#ced0ec",
        "#6a2d0a",
        "#9db5ea",
        "#a35c1b",
        "#4781b1",
        "#9e4e22",
        "#33547a",
        "#876a1c",
        "#514e80",
        "#a59952",
        "#b86198",
        "#1d3621",
        "#eb7ba2",
        "#002a33",
        "#e38273",
        "#17212e",
        "#e8c4c5",
        "#281c2e",
        "#b3b18a",
        "#581430",
        "#659c84",
        "#a23a50",
        "#2d7681",
        "#a44634",
        "#608ea2",
        "#783121",
        "#94a9bc",
        "#4b1615",
        "#a4ae9f",
        "#7c3258",
        "#aa8242",
        "#7a6ea2",
        "#5f5621",
        "#c27dae",
        "#403911",
        "#a499c7",
        "#805124",
        "#717e9e",
        "#b8644f",
        "#143b44",
        "#ce6472",
        "#142a25",
        "#dd9ca6",
        "#21344a",
        "#d7a78c",
        "#3c3551",
        "#928853",
        "#ad486c",
        "#3a4d2d",
        "#8c5481",
        "#516b4d",
        "#994440",
        "#2e5667",
        "#af7e5c",
        "#432432",
        "#b49bb0",
        "#382718",
        "#b67576",
        "#294d46",
        "#935c54",
        "#52756e",
        "#6d363c",
        "#85856a",
        "#644466",
        "#635738",
        "#876d84",
        "#623c23",
        "#596776",
        "#864e5d",
        "#5f5848",
        "#9f7e80",
        "#5c4a56",
        "#735647",
        "#bcbcbc",
    ]

    return colours_w_old_colors
