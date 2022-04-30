from napari import Viewer
from napari.types import LabelsData
from napari_tools_menu import register_function
from skimage.measure import regionprops_table
import numpy as np
import warnings

from ._utilities import add_column_to_layer_tabular_data


@register_function(menu="Measurement > Convert annotation to cluster ID (ncp)")
def Annotation_to_Cluster_ID(
    label_image: LabelsData, 
    annotation: LabelsData, 
    Unannotated_Objects_Clustered: bool = True, 
    viewer: Viewer = None
) -> None:
    print(label_image.shape)
    if len(label_image.shape) <= 3:
        regionproperties = regionprops_table(
            label_image, intensity_image=annotation, properties=("label", "intensity_max")
        )
        intensities = regionproperties['intensity_max']
    elif len(label_image.shape) == 4:
        regp_list = [
            regionprops_table(
                label, intensity_image=anno, properties=("label", "intensity_max")
            ) 
            for label,anno in zip(label_image,annotation)
        ]
        intensities = np.concatenate([regp['intensity_max'] for regp in regp_list])
    else:
        warnings.warn("Image dimensions too high for processing!")
        return


    if Unannotated_Objects_Clustered:
        data = intensities
    else:
        data = intensities-1
    

    if viewer is not None:
        # store the layer for saving results later
        from napari_workflows._workflow import _get_layer_from_data

        labels_layer = _get_layer_from_data(viewer, label_image)

        # Store results in the properties dictionary:
        add_column_to_layer_tabular_data(
            layer=labels_layer, column_name="Annotation_CLUSTER_ID", data=data
        )

        # turn table into a widget
        from napari_skimage_regionprops import add_table

        add_table(labels_layer, viewer)
    else:
        return data
