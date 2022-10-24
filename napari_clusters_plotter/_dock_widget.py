from napari_plugin_engine import napari_hook_implementation

from ._clustering import ClusteringWidget
from ._dimensionality_reduction import DimensionalityReductionWidget
from ._plotter import PlotterWidget
from ._import_measurements import ImportWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        ImportWidget,
        PlotterWidget,
        DimensionalityReductionWidget,
        ClusteringWidget,
    ]
