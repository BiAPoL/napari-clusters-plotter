from napari import Viewer
from napari.types import LabelsData
from napari_tools_menu import register_function
from skimage.measure import regionprops_table
import numpy as np
import warnings
import dask.array as da

from ._utilities import add_column_to_layer_tabular_data


ANNOTATION_ID = "Annotation_CLUSTER_ID"

@register_function(menu="Measurement > Convert annotation to cluster ID (ncp)")
def Annotation_to_Cluster_ID(
    label_image: LabelsData, 
    annotation: LabelsData, 
    viewer: Viewer = None
) -> None:
    print(label_image.shape)
    if len(label_image.shape) <= 3:
        if isinstance(label_image, da.core.Array):
            label_image = np.asarray(label_image)
        if isinstance(annotation, da.core.Array):
            annotation = np.asanyarray(annotation)
        regionproperties = regionprops_table(
            label_image, intensity_image=annotation, properties=("label", "intensity_max")
        )
        intensities = regionproperties['intensity_max']

    elif len(label_image.shape) == 4:
        regp_list = []
        for label,anno in zip(label_image,annotation):
            if isinstance(label, da.core.Array):
                label = np.asarray(label)
            if isinstance(anno, da.core.Array):
                anno = np.asanyarray(anno)

            regp_list.append(
                regionprops_table(
                    label, intensity_image=anno, properties=("label", "intensity_max")
                )['intensity_max'].astype('uint16')
            )
        intensities = np.concatenate(regp_list)
    else:
        warnings.warn("Image dimensions too high for processing!")
        return

    if viewer is not None:
        # store the layer for saving results later
        from napari_workflows._workflow import _get_layer_from_data

        labels_layer = _get_layer_from_data(viewer, label_image)

        # Store results in the properties dictionary:
        add_column_to_layer_tabular_data(
            layer=labels_layer, column_name=ANNOTATION_ID, data=intensities   
        )

        # turn table into a widget
        from napari_skimage_regionprops import add_table

        add_table(labels_layer, viewer)
    else:
        return intensities
