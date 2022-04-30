from napari_plugin_engine import napari_hook_implementation

from ._annotation_to_clusters import annotation_to_cluster_id
from ._clustering import ClusteringWidget
from ._dimensionality_reduction import DimensionalityReductionWidget
from ._measure import MeasureWidget
from ._plotter import PlotterWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        MeasureWidget,
        PlotterWidget,
        DimensionalityReductionWidget,
        ClusteringWidget,
        annotation_to_cluster_id,
    ]
