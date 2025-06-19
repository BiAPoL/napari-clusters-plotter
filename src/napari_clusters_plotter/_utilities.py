from typing import Union

import dask.array as da
import numpy as np
from napari.layers import Image, Labels


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
