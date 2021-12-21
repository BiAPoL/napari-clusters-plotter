from napari_plugin_engine import napari_hook_implementation
from ._measure import MeasureWidget
from ._plotter import PlotterWidget
from ._dimensionality_reduction import DimensionalityReductionWidget
from ._kmeans_clustering import ClusteringWidget
from ._feature_selection import FeatureSelectionWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [MeasureWidget, PlotterWidget, DimensionalityReductionWidget, ClusteringWidget, FeatureSelectionWidget]
