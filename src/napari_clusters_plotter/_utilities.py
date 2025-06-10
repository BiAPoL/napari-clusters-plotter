import numpy as np
import dask.array as da
from typing import Union


def _get_unique_values(array: Union[da.Array, np.ndarray]) -> np.ndarray:
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

    if isinstance(array, np.ndarray):
        unique_values = np.unique(array)
    elif isinstance(array, da.Array):
        unique_values = da.unique(array).compute()

    return unique_values
