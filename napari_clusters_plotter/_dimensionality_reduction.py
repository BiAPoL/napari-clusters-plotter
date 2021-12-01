import pandas as pd
import warnings
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QComboBox, QSpinBox
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive
from napari_tools_menu import register_dock_widget


@register_dock_widget(menu="Measurement > Dimensionality reduction (ncp)")
class DimensionalityReductionWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Dimensionality reduction</b>"))

        # selection of labels layer
        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())
        self.label_list = QComboBox()
        self.update_label_list()
        self.label_list.currentIndexChanged.connect(self.update_properties_list)
        choose_img_container.layout().addWidget(QLabel("Labels layer"))
        choose_img_container.layout().addWidget(self.label_list)

        # selection of dimension reduction algorithm
        algorithm_container = QWidget()
        algorithm_container.setLayout(QHBoxLayout())
        algorithm_container.layout().addWidget(QLabel("Dimensionality Reduction Algorithm"))
        self.algorithm_choice_list = QComboBox()
        self.algorithm_choice_list.addItems(['   ', 'UMAP', 't-SNE'])
        algorithm_container.layout().addWidget(self.algorithm_choice_list)

        # selection of n_neighbors - The size of local neighborhood (in terms of number of neighboring sample points)
        # used for manifold approximation. Larger values result in more global views of the manifold, while smaller
        # values result in more local data being preserved.
        self.n_neighbors_container = QWidget()
        self.n_neighbors_container.setLayout(QHBoxLayout())
        self.n_neighbors_container.layout().addWidget(QLabel("Number of neighbors"))
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setMinimumWidth(40)
        self.n_neighbors.setMinimum(2)
        self.n_neighbors.setValue(15)
        self.n_neighbors_container.layout().addWidget(self.n_neighbors)
        self.n_neighbors_container.setVisible(False)

        # selection of the level of perplexity. Higher values should be chosen when handling large datasets
        self.perplexity_container = QWidget()
        self.perplexity_container.setLayout(QHBoxLayout())
        self.perplexity_container.layout().addWidget(QLabel("Perplexity"))
        self.perplexity = QSpinBox()
        self.perplexity.setMinimumWidth(40)
        self.perplexity.setMinimum(1)
        self.perplexity.setValue(30)
        self.perplexity_container.layout().addWidget(self.perplexity)
        self.perplexity_container.setVisible(False)

        # select properties of which to produce a dimension reduce version
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
                [i.text() for i in self.properties_list.selectedItems()],
                self.n_neighbors.value(), self.perplexity.value(),
                self.algorithm_choice_list.currentText()
                # todo: enter number of components here as third parameter
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(algorithm_container)
        self.layout().addWidget(self.perplexity_container)
        self.layout().addWidget(self.n_neighbors_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_neighbours_list)
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_perplexity)

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
    def run(self, labels_layer, selected_measurements_list, n_neighbours, perplexity, selected_algorithm,
            n_components=2):
        print("Dimensionality reduction running")
        print(labels_layer)
        print(selected_measurements_list)

        # Turn properties from layer into a dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        properties_to_reduce = reg_props[selected_measurements_list]

        if selected_algorithm == 'UMAP':
            # reduce dimensions
            embedding = umap(properties_to_reduce, n_neighbours, n_components)

            # write result back to properties
            for i in range(0, n_components):
                properties["UMAP_" + str(i)] = embedding[:, i]

        elif selected_algorithm == 't-SNE':
            # reduce dimensions
            embedding = tsne(properties_to_reduce, perplexity, n_components)

            # write result back to properties
            for i in range(0, n_components):
                properties['t-SNE_' + str(i)] = embedding[:, i]

        else:
            warnings.warn('No Dimension Reduction Algorithm Chosen!')

        from ._utilities import show_table
        show_table(self.viewer, labels_layer)

        print("Dimensionality reduction finished")

    # toggle widgets visibility according to what is selected
    def change_neighbours_list(self):
        widgets_inactive(self.n_neighbors_container, active=self.algorithm_choice_list.currentText() == 'UMAP')

    def change_perplexity(self):
        widgets_inactive(self.perplexity_container, active=self.algorithm_choice_list.currentText() == 'TSNE')


def umap(reg_props, n_neigh=15, n_components=2):
    from sklearn.preprocessing import StandardScaler
    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=133, n_components=n_components, n_neighbors=n_neigh)

    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    return reducer.fit_transform(scaled_regionprops)


def tsne(reg_props, perplexity, n_components=2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    reducer = TSNE(perplexity=perplexity, n_components=n_components, learning_rate='auto', init='pca', random_state=42)

    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    return reducer.fit_transform(scaled_regionprops)
