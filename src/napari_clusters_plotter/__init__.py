__version__ = "0.8.0"

from ._dim_reduction_and_clustering import (
    ClusteringWidget,
    DimensionalityReductionWidget,
)
from ._new_plotter_widget import PlotterWidget

__all__ = [
    "PlotterWidget",
    "DimensionalityReductionWidget",
    "ClusteringWidget",
]
