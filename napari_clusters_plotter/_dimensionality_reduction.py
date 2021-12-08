from functools import partial

import pandas as pd
import warnings
from napari.layers import Labels
from magicgui.widgets import create_widget
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive, restore_defaults
from napari_tools_menu import register_dock_widget

# Remove when the problem is fixed from sklearn side
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

DEFAULTS = dict(
    n_neighbors=15,
    perplexity=30,
)


@register_dock_widget(menu="Measurement > Dimensionality reduction (ncp)")
class DimensionalityReductionWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Dimensionality reduction</b>"))

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # selection of dimension reduction algorithm
        algorithm_container = QWidget()
        algorithm_container.setLayout(QHBoxLayout())
        algorithm_container.layout().addWidget(QLabel("Dimensionality Reduction Algorithm"))
        self.algorithm_choice_list = QComboBox()
        self.algorithm_choice_list.addItems(['', 'UMAP', 't-SNE'])
        algorithm_container.layout().addWidget(self.algorithm_choice_list)

        # selection of n_neighbors - The size of local neighborhood (in terms of number of neighboring sample points)
        # used for manifold approximation. Larger values result in more global views of the manifold, while smaller
        # values result in more local data being preserved.
        self.n_neighbors_container = QWidget()
        self.n_neighbors_container.setLayout(QHBoxLayout())
        self.n_neighbors_container.layout().addWidget(QLabel("Number of neighbors"))
        self.n_neighbors = create_widget(widget_type="SpinBox",
                                         name='n_neighbors',
                                         value=DEFAULTS['n_neighbors'],
                                         options=dict(min=2, step=1))

        self.n_neighbors_container.layout().addWidget(self.n_neighbors.native)
        self.n_neighbors_container.setVisible(False)

        # selection of the level of perplexity. Higher values should be chosen when handling large datasets
        self.perplexity_container = QWidget()
        self.perplexity_container.setLayout(QHBoxLayout())
        self.perplexity_container.layout().addWidget(QLabel("Perplexity"))
        self.perplexity = create_widget(widget_type="SpinBox",
                                        name='perplexity',
                                        value=DEFAULTS['perplexity'],
                                        options=dict(min=1, step=1))

        self.perplexity_container.layout().addWidget(self.perplexity.native)
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
        run_button = QPushButton("Run")
        run_widget.layout().addWidget(run_button)

        # Update measurements button
        update_container = QWidget()
        update_container.setLayout(QHBoxLayout())
        update_button = QPushButton("Update Measurements")
        update_container.layout().addWidget(update_button)

        # Defaults button
        defaults_container = QWidget()
        defaults_container.setLayout(QHBoxLayout())
        defaults_button = QPushButton("Restore Defaults")
        defaults_container.layout().addWidget(defaults_button)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some measurements!")
                return

            if self.algorithm_choice_list.currentText() == "":
                warnings.warn("Please select dimensionality reduction algorithm.")
                return

            self.run(
                self.labels_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                self.n_neighbors.value, self.perplexity.value,
                self.algorithm_choice_list.currentText()
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(algorithm_container)
        self.layout().addWidget(self.perplexity_container)
        self.layout().addWidget(self.n_neighbors_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(defaults_container)
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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_neighbours_list(self):
        widgets_inactive(self.n_neighbors_container, active=self.algorithm_choice_list.currentText() == 'UMAP')

    def change_perplexity(self):
        widgets_inactive(self.perplexity_container, active=self.algorithm_choice_list.currentText() == 't-SNE')

    def update_properties_list(self):
        selected_layer = self.labels_select.value
        if selected_layer is not None:
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                self.properties_list.clear()
                for p in list(properties.keys()):
                    if p == "label" or "CLUSTER_ID" in p:
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, n_neighbours, perplexity, selected_algorithm,
            n_components=2):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        # Turn properties from layer into a dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        properties_to_reduce = reg_props[selected_measurements_list]

        if selected_algorithm == 'UMAP':
            print("Dimensionality reduction started (" + str(selected_algorithm) + ")...")
            # reduce dimensionality
            embedding = umap(properties_to_reduce, n_neighbours, n_components)

            # write result back to properties
            for i in range(0, n_components):
                properties["UMAP_" + str(i)] = embedding[:, i]

        elif selected_algorithm == 't-SNE':
            print("Dimensionality reduction started (" + str(selected_algorithm) + ")...")
            # reduce dimensionality
            embedding = tsne(properties_to_reduce, perplexity, n_components)

            # write result back to properties
            for i in range(0, n_components):
                properties['t-SNE_' + str(i)] = embedding[:, i]

        from ._utilities import show_table
        show_table(self.viewer, labels_layer)

        print("Dimensionality reduction finished")


def umap(reg_props, n_neigh, n_components):  # n_components: dimension of the embedded space. For now 2 by default,
    from sklearn.preprocessing import StandardScaler  # since only 2D plotting is supported
    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=133, n_components=n_components, n_neighbors=n_neigh)

    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    return reducer.fit_transform(scaled_regionprops)


def tsne(reg_props, perplexity, n_components):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    reducer = TSNE(perplexity=perplexity, n_components=n_components, learning_rate='auto', init='pca', random_state=42)

    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    return reducer.fit_transform(scaled_regionprops)
