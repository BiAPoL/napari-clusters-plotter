from napari_plugin_engine import napari_hook_implementation

from ._clustering import ClusteringWidget
from ._dimensionality_reduction_new import DimensionalityReductionWidget
from ._new_plotter_widget import PlotterWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        PlotterWidget,
        DimensionalityReductionWidget,
        ClusteringWidget,
    ]
