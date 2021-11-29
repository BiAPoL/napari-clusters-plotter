import pyclesperanto_prototype as cle
from pandas import DataFrame
import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QGridLayout, QFileDialog, QTableWidget, QTableWidgetItem
from qtpy.QtCore import QTimer
import warnings


def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


def show_table(viewer, labels_layer):
    from napari_skimage_regionprops import add_table
    add_table(labels_layer, viewer)


# function for generating image labelled by clusters given the label image and the cluster prediction list
def generate_parametric_cluster_image(labelimage, predictionlist):
    print('Generation of parametric cluster image started')

    # reforming the prediction list this is done to account for cluster labels that start at 0
    # conveniently hdbscan labelling starts at -1 for noise, removing these from the labels
    predictionlist_new = np.array(predictionlist) + 1

    # this takes care of the background label that needs to be 0 as well as any other
    # labels that might have been accidentally deleted
    for i in range(int(np.min(labelimage[np.nonzero(labelimage)]))):
        predictionlist_new = np.insert(predictionlist_new, i, 0)

    # pushing of variables into GPU
    cle_list = cle.push(predictionlist_new)
    cle_labels = cle.push(labelimage)

    # generation of cluster label image
    parametric_image = cle.pull(cle.replace_intensities(cle_labels, cle_list))
    print('Generation of parametric cluster image finished')
    return np.array(parametric_image, dtype="int64")
