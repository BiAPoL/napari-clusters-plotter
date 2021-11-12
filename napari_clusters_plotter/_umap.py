import pandas as pd
import warnings
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
from qtpy.QtCore import QRect
from napari_tools_menu import  register_dock_widget

@register_dock_widget(menu="Measurement > UMAP dimensionality reduction (ncp)")
class UMAPWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # setup layout of the whole dialog. QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Dimensionality reduction: UMAP</b>"))

        # selection of labels layer
        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())
        self.label_list = QComboBox()
        self.update_label_list()
        self.label_list.currentIndexChanged.connect(self.update_properties_list)
        choose_img_container.layout().addWidget(QLabel("Labels layer"))
        choose_img_container.layout().addWidget(self.label_list)

        # select properties to make a umap from
        choose_properties_container = QWidget()
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))
        self.update_properties_list()

        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Measurements"))
        choose_properties_container.layout().addWidget(self.properties_list)

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        def run_clicked():
            if self.get_selected_label() is None:
                warnings.warn("No labels image was selected!")
                return

            self.run(
                self.get_selected_label(),
                [i.text() for i in self.properties_list.selectedItems()]
                # todo: enter number of components here as third parameter
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

    def get_selected_label(self):
        index = self.label_list.currentIndex()
        if index >= 0:
            return self._available_labels[index]
        return None

    def update_label_list(self):
        selected_layer = self.get_selected_label()
        selected_index = -1

        self._available_labels = []
        self.label_list.clear()
        i = 0
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._available_labels.append(layer)
                if layer == selected_layer:
                    selected_index = i
                self.label_list.addItem(layer.name)
                i = i + 1
        self.label_list.setCurrentIndex(selected_index)

    def update_properties_list(self):
        selected_layer = self.get_selected_label()

        print("Selected layer none?")
        if selected_layer is not None:
            print("Properties none?")
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                self.properties_list.clear()
                for p in list(properties.keys()):
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    # per default select all measurements that are not "label
                    if p != "label":
                        item.setSelected(True)

    def _on_selection(self, event=None):
        num_labels_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, n_components=2):
        print("Dimensionality reduction running")
        print(labels_layer)
        print(selected_measurements_list)

        # Turn properties from layer into a dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        properties_to_reduce = reg_props[selected_measurements_list]

        # reduce dimensions
        embedding = umap(properties_to_reduce, n_components)

        # write result back to properties
        for i in range(0, n_components):
            properties["UMAP_" + str(i)] = embedding[:,i]

        from ._utilities import show_table
        show_table(self.viewer, labels_layer)

        print("Dimensionality reduction finished")

def umap(reg_props, n_components=2):
    from sklearn.preprocessing import StandardScaler
    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=133, n_components=n_components)
    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    return reducer.fit_transform(scaled_regionprops)
