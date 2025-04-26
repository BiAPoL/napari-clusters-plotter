__version__ = "0.8.0"

from ._dim_reduction_and_clustering import (
    ClusteringWidget,
    DimensionalityReductionWidget,
)
from ._new_plotter_widget import PlotterWidget
from ._sample_data import bbbc_1_dataset, cells3d_curvatures

__all__ = [
    "PlotterWidget",
    "DimensionalityReductionWidget",
    "ClusteringWidget",
    "bbbc_1_dataset",
    "cells3d_curvatures",
]
