from napari_plugin_engine import napari_hook_implementation
from ._measure import MeasureWidget
from ._plotter import PlotterWidget
from ._umap import UMAPWidget
from ._tsne import TSNEWidget
from ._kmeans_clustering import ClusteringWidget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [MeasureWidget, PlotterWidget, UMAPWidget, TSNEWidget, ClusteringWidget]
