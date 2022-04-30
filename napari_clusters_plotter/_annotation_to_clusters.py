from ._utilities import add_column_to_layer_tabular_data, show_table
from skimage.measure import regionprops_table
from napari_clusters_plotter._utilities import show_table
from napari.layers import Labels
from napari import Viewer
from napari_tools_menu import register_function


@register_function(
    menu="Measurement > Convert annotation to cluster ID (ncp)"
)
def annotation_to_cluster_id(Label_Layer: Labels = None, Annotation_Layer: Labels = None, viewer: Viewer = None, Unannotated_Objects_Clustered: bool = False) -> None:
    label_image = Label_Layer.data
    annotation = Annotation_Layer.data
    regionproperties = regionprops_table(label_image, intensity_image=annotation, properties=('label', 'intensity_max'))
    if Unannotated_Objects_Clustered:
        data = regionproperties['intensity_max']
    else:
        data = regionproperties['intensity_max']-1
    
    add_column_to_layer_tabular_data(
        layer=Label_Layer,
        column_name="Annotation_CLUSTER_ID",
        data=data
    )
    show_table(labels_layer=Label_Layer, viewer=viewer)