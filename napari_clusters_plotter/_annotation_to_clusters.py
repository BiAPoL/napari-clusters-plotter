from ._utilities import add_column_to_layer_tabular_data
from skimage.measure import regionprops_table
from napari.types import LabelsData
from napari import Viewer
from napari_tools_menu import register_function


@register_function(
    menu="Measurement > Convert annotation to cluster ID (ncp)"
)
def annotation_to_cluster_id(label_image: LabelsData, annotation: LabelsData, viewer: Viewer = None) -> None:
    regionproperties = regionprops_table(label_image, intensity_image=annotation, properties=('label', 'intensity_max'))
    data = regionproperties['intensity_max']

    if viewer is not None:
        # store the layer for saving results later
        from napari_workflows._workflow import _get_layer_from_data
        labels_layer = _get_layer_from_data(viewer, label_image)

        # Store results in the properties dictionary:
        add_column_to_layer_tabular_data(
            layer=labels_layer,
            column_name="Annotation_CLUSTER_ID",
            data=data
        )

        # turn table into a widget
        from napari_skimage_regionprops import add_table
        add_table(labels_layer, viewer)
    else:
        return data
