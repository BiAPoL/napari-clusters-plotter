import pyclesperanto_prototype as cle
import pandas as pd
import numpy as np
import warnings
# import hdbscan
import napari
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from napari_plugin_engine import napari_hook_implementation
from PyQt5 import QtWidgets
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QSpinBox, QHBoxLayout, QVBoxLayout, QComboBox, QGridLayout, \
    QFileDialog, QTableWidget, QTableWidgetItem
from qtpy.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib.patches import Rectangle


def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)

def show_table(viewer, labels_layer):

    dock_widget = _table_to_widget(labels_layer.properties, labels_layer)
    viewer.window.add_dock_widget(dock_widget, name='Region properties table', area='right')

# from Robert Haase napari-skimage-regionprops
def _table_to_widget(table: dict, labels_layer: napari.layers.Labels) -> QWidget:
    """
    Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
    contains a QTableWidget with that data.
    """
    view = QTableWidget(len(next(iter(table.values()))), len(table))
    for i, column in enumerate(table.keys()):
        view.setItem(0, i, QTableWidgetItem(column))
        for j, value in enumerate(table.get(column)):
            view.setItem(j + 1, i, QTableWidgetItem(str(value)))

    if labels_layer is not None:

        @view.clicked.connect
        def clicked_table():
            row = view.currentRow()
            label = table["label"][row]
            labels_layer.selected_label = label

        def after_labels_clicked():
            row = view.currentRow()
            label = table["label"][row]
            if label != labels_layer.selected_label:
                for r, layer in enumerate(table["label"]):
                    if layer == labels_layer.selected_label:
                        view.setCurrentCell(r, view.currentColumn())
                        break

        @labels_layer.mouse_drag_callbacks.append
        def clicked_labels(event, event1):
            # We need to run this lagter as the labels_layer.selected_label isn't changed yet.
            QTimer.singleShot(200, after_labels_clicked)

    copy_button = QPushButton("Copy to clipboard")

    @copy_button.clicked.connect
    def copy_trigger():
        view.to_dataframe().to_clipboard()

    save_button = QPushButton("Save as csv...")

    @save_button.clicked.connect
    def save_trigger():
        filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
        view.to_dataframe().to_csv(filename)

    widget_table = QWidget()
    widget_table.setWindowTitle("region properties")
    widget_table.setLayout(QGridLayout())
    widget_table.layout().addWidget(copy_button)
    widget_table.layout().addWidget(save_button)
    widget_table.layout().addWidget(view)

    return widget_table
