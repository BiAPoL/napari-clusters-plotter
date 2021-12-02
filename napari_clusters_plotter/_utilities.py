import pyclesperanto_prototype as cle
import numpy as np


def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


def show_table(viewer, labels_layer):
    from napari_skimage_regionprops import add_table
    add_table(labels_layer, viewer)


# function for generating image labelled by clusters given the label image and the cluster prediction list
def generate_parametric_cluster_image(label_image, prediction_list):
    print('Generation of parametric cluster image started')

    # reforming the prediction list; this is done to account for cluster labels that start at 0
    prediction_list_new = np.array(prediction_list) + 1

    # this takes care of the background label that needs to be 0 as well as any other
    # labels that might have been accidentally deleted
    for i in range(int(np.min(label_image[np.nonzero(label_image)]))):
        prediction_list_new = np.insert(prediction_list_new, i, 0)

    # pushing of variables into GPU
    cle_list = cle.push(prediction_list_new)
    cle_labels = cle.push(label_image)

    # generation of cluster label image
    parametric_image = cle.pull(cle.replace_intensities(cle_labels, cle_list))
    print('Generation of parametric cluster image finished')
    return np.array(parametric_image, dtype="int64")


def restore_defaults(widget, defaults: dict):
    for item, val in defaults.items():
        getattr(widget, item).value = val
