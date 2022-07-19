from qtpy.QtCore import QRect
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from magicgui.widgets import create_widget
from napari.layers import Labels

def measurements_container_and_list():
    properties_container = QWidget()
    properties_container.setLayout(QVBoxLayout())
    properties_container.layout().addWidget(QLabel("Measurements"))
    properties_list = QListWidget()
    properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    properties_list.setGeometry(QRect(10, 10, 101, 291))
    properties_container.layout().addWidget(properties_list)

    return properties_container, properties_list

def labels_selction_container_and_selection():
    labels_layer_selection_container = QWidget()
    labels_layer_selection_container.setLayout(QHBoxLayout())
    labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
    labels_select = create_widget(annotation=Labels, label="labels_layer")
    labels_layer_selection_container.layout().addWidget(labels_select.native)

    return labels_layer_selection_container, labels_select