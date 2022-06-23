from napari_plugin_engine import napari_hook_implementation

from ._clustering import ClusteringWidget
from ._dimensionality_reduction import DimensionalityReductionWidget
from ._feature_selection import FeatureSelectionWidget
from ._measure import MeasureWidget
from ._plotter import PlotterWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        MeasureWidget,
        PlotterWidget,
        DimensionalityReductionWidget,
        ClusteringWidget,
        FeatureSelectionWidget,
    ]
