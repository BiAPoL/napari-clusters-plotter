try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._dim_reduction_and_clustering import (
    ClusteringWidget,
    DimensionalityReductionWidget,
)
from ._new_plotter_widget import PlotterWidget
from ._sample_data import (
    bbbc_1_dataset,
    cells3d_curvatures,
    skan_skeleton,
    tgmm_mini_dataset,
)

__all__ = [
    "PlotterWidget",
    "DimensionalityReductionWidget",
    "ClusteringWidget",
    "bbbc_1_dataset",
    "tgmm_mini_dataset",
    "cells3d_curvatures",
    "skan_skeleton",
]
