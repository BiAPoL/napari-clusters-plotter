from typing import Union, List

import dask.array as da
import numpy as np
from napari.layers import Image, Labels, Points, Shapes, Layer
from napari.utils.events import Event

_selectable_layers = [
    Labels,
    Points,
#    Shapes
]


def _get_unique_values(layer: Union[Image, Labels]) -> np.ndarray:
    """
    Get unique values from a numpy or dask array.

    Parameters
    ----------
    array : np.ndarray or dask.array.Array
        Input array from which to extract unique values.

    Returns
    -------
    np.ndarray
        Array of unique values.
    """
    # Check if the layer is multiscale and extract the first scale data
    # TODO: Discuss whether this is a smart thing to do....
    data = layer.data[0] if layer.multiscale else layer.data

    if isinstance(data, np.ndarray):
        unique_values = np.unique(data)
    elif isinstance(data, da.Array):
        unique_values = da.unique(data).compute()

    return unique_values


def _is_selectable_layer(layer: Layer) -> bool:
    """
    Check if the layer is selectable.
    """
    if type(layer) in _selectable_layers:
        return True
    return False


def _get_selected_objects(layer: Layer) -> List[int]:
    """
    Retrieve id of selected object on napari canvas
    """
    if not _is_selectable_layer(layer):
        raise TypeError(
            f"Layer type {type(layer)} is not supported for selection."
        )

    if isinstance(layer, Points):
        return list(layer.selected_data)
    elif isinstance(layer, Labels):
        return [layer.selected_label]
    elif isinstance(layer, Shapes):
        return list(layer.selected_data)


def _get_selection_event(
    layer: Layer
) -> Event:
    """
    Get the selection event for the layer.
    """
    if not _is_selectable_layer(layer):
        raise TypeError(
            f"Layer type {type(layer)} is not supported for selection events."
        )
    if isinstance(layer, Points):
        return layer.selected_data.events.items_changed
    elif isinstance(layer, Labels):
        return layer.events.selected_label
    elif isinstance(layer, Shapes):
        return layer.events.highlight