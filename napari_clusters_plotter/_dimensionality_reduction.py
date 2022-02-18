from functools import partial
import warnings
from napari.layers import Labels
from magicgui.widgets import create_widget
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive, restore_defaults, get_layer_tabular_data, \
    add_column_to_layer_tabular_data
from napari_tools_menu import register_dock_widget

# Remove when the problem is fixed from sklearn side
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

DEFAULTS = {
    "n_neighbors": 15,
    "perplexity": 30,
    "standardization": True,
    "pca_components": 0,
    "explained_variance": 95.0
}


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
        self.algorithm_choice_list.addItems(['', 'UMAP', 't-SNE', 'PCA'])
        algorithm_container.layout().addWidget(self.algorithm_choice_list)

        # selection of n_neighbors - The size of local neighborhood (in terms of number of neighboring sample points)
        # used for manifold approximation. Larger values result in more global views of the manifold, while smaller
        # values result in more local data being preserved.
        self.n_neighbors_container = QWidget()
        self.n_neighbors_container.setLayout(QHBoxLayout())
        self.n_neighbors_container.layout().addWidget(QLabel("Number of neighbors"))
        self.n_neighbors_container.layout().addStretch()
        self.n_neighbors = create_widget(widget_type="SpinBox",
                                         name='n_neighbors',
                                         value=DEFAULTS['n_neighbors'],
                                         options=dict(min=2, step=1))

        help_n_neighbors = QLabel()
        help_n_neighbors.setOpenExternalLinks(True)
        help_n_neighbors.setText('<a href="https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors" '
                                 'style="text-decoration:none; color:white"><b>?</b></a>')

        help_n_neighbors.setToolTip(
            "The size of local neighborhood (in terms of number of neighboring sample points) used for manifold "
            "approximation. Larger values result in more global views of the manifold, while smaller values should be "
            "in the range 2 to 100. Click on the question mark to read more.")

        self.n_neighbors.native.setMaximumWidth(70)
        self.n_neighbors_container.layout().addWidget(self.n_neighbors.native)
        self.n_neighbors_container.layout().addWidget(help_n_neighbors)
        self.n_neighbors_container.setVisible(False)

        # selection of the level of perplexity. Higher values should be chosen when handling large datasets
        self.perplexity_container = QWidget()
        self.perplexity_container.setLayout(QHBoxLayout())
        self.perplexity_container.layout().addWidget(QLabel("Perplexity"))
        self.perplexity_container.layout().addStretch()
        self.perplexity = create_widget(widget_type="SpinBox",
                                        name='perplexity',
                                        value=DEFAULTS['perplexity'],
                                        options=dict(min=1, step=1))

        help_perplexity = QLabel()
        help_perplexity.setOpenExternalLinks(True)
        help_perplexity.setText('<a href="https://distill.pub/2016/misread-tsne/" '
                                'style="text-decoration:none; color:white"><b>?</b></a>')

        help_perplexity.setToolTip(
            "The perplexity is related to the number of nearest neighbors that is used in other manifold learning "
            "algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and "
            "50. Different values can result in significantly different results. "
            "Click on the question mark to read more.")

        self.perplexity.native.setMaximumWidth(70)
        self.perplexity_container.layout().addWidget(self.perplexity.native)
        self.perplexity_container.layout().addWidget(help_perplexity)
        self.perplexity_container.setVisible(False)

        # select properties of which to produce a dimensionality reduced version
        choose_properties_container = QWidget()
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))

        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Measurements"))
        choose_properties_container.layout().addWidget(self.properties_list)

        # selection of the number of components to keep after PCA transformation,
        # values above 0 will override explained variance option
        self.pca_components_container = QWidget()
        self.pca_components_container.setLayout(QHBoxLayout())
        self.pca_components_container.layout().addWidget(QLabel("Number of Components"))
        self.pca_components = create_widget(widget_type="SpinBox",
                                        name='pca_components',
                                        value=DEFAULTS['pca_components'],
                                        options=dict(min=0, step=1)) #TODO , max=len(self.properties_list)

        self.pca_components_container.layout().addWidget(self.pca_components.native)
        self.pca_components_container.setVisible(False)

        # Minimum percentage of variance explained by kept PCA components,
        # will not be used if pca_components > 0 
        self.explained_variance_container = QWidget()
        self.explained_variance_container.setLayout(QHBoxLayout())
        self.explained_variance_container.layout().addWidget(QLabel("Explained Variance Threshold"))
        self.explained_variance = create_widget(widget_type="FloatSpinBox",
                                        name='explained_variance',
                                        value=DEFAULTS['explained_variance'],
                                        options=dict(min=1, max=100, step=1))

        self.explained_variance_container.layout().addWidget(self.explained_variance.native)
        self.explained_variance_container.setVisible(False)
        # checkbox whether data should be standardized
        self.settings_container_scaler = QWidget()
        self.settings_container_scaler.setLayout(QHBoxLayout())
        self.standardization = create_widget(widget_type="CheckBox", name="Standardize Features",
                                             value=DEFAULTS["standardization"])

        self.settings_container_scaler.layout().addWidget(self.standardization.native)
        self.settings_container_scaler.setVisible(False)

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

            self.run(self.labels_select.value,
                     [i.text() for i in self.properties_list.selectedItems()],
                     self.n_neighbors.value, self.perplexity.value,
                     self.algorithm_choice_list.currentText(),
                     self.standardization.value,
                     self.explained_variance.value,
                     self.pca_components.value)

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(self.update_properties_list)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(algorithm_container)
        self.layout().addWidget(self.perplexity_container)
        self.layout().addWidget(self.n_neighbors_container)
        self.layout().addWidget(self.pca_components_container)
        self.layout().addWidget(self.explained_variance_container)
        self.layout().addWidget(self.settings_container_scaler)
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
        #self.algorithm_choice_list.currentIndexChanged.connect(self.change_neighbours_list)
        #self.algorithm_choice_list.currentIndexChanged.connect(self.change_perplexity)
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_pca_components)
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_explained_variance)
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_umap_settings)
        self.algorithm_choice_list.currentIndexChanged.connect(self.change_tsne_settings)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_umap_settings(self):
        widgets_inactive(self.n_neighbors_container, active=self.algorithm_choice_list.currentText() == 'UMAP')
        widgets_inactive(self.settings_container_scaler,
                         active=(self.algorithm_choice_list.currentText() == 'UMAP' or
                                 self.algorithm_choice_list.currentText() == 't-SNE'))

    def change_tsne_settings(self):
        widgets_inactive(self.perplexity_container, active=self.algorithm_choice_list.currentText() == 't-SNE')
        widgets_inactive(self.settings_container_scaler,
                         active=(self.algorithm_choice_list.currentText() == 'UMAP' or
                                 self.algorithm_choice_list.currentText() == 't-SNE'))

    def change_pca_components(self):
        widgets_inactive(self.pca_components_container, active=self.algorithm_choice_list.currentText() == 'PCA')

    def change_explained_variance(self):
        widgets_inactive(self.explained_variance_container, active=self.algorithm_choice_list.currentText() == 'PCA')



    def update_properties_list(self):
        selected_layer = self.labels_select.value
        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if "label" in p or "CLUSTER_ID" in p or "UMAP" in p or "t-SNE" in p:
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, n_neighbours, perplexity, selected_algorithm, standardize,
            explained_variance, pca_components, n_components=2):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        features = get_layer_tabular_data(labels_layer)

        # only select the columns the user requested
        properties_to_reduce = features[selected_measurements_list]

        if selected_algorithm == 'UMAP':
            print("Dimensionality reduction started (" + str(selected_algorithm) + ", standardize: " + str(standardize)
                  + ")...")
            # reduce dimensionality
            embedding = umap(properties_to_reduce, n_neighbours, n_components, standardize)

            # write result back to features/properties
            for i in range(0, n_components):
                add_column_to_layer_tabular_data(labels_layer, "UMAP_" + str(i), embedding[:, i])

        elif selected_algorithm == 't-SNE':
            print("Dimensionality reduction started (" + str(selected_algorithm) + ", standardize: " + str(standardize)
                  + ")...")
            # reduce dimensionality
            embedding = tsne(properties_to_reduce, perplexity, n_components, standardize)

            # write result back to features/properties
            for i in range(0, n_components):
                add_column_to_layer_tabular_data(labels_layer, "t-SNE_" + str(i), embedding[:, i])
        
        elif selected_algorithm == 'PCA':
            print("Dimensionality reduction started (" + str(selected_algorithm) + ")...")
            # reduce dimensionality
            embedding = pca(properties_to_reduce, explained_variance, pca_components)

            # write result back to properties
            for i in range(0, len(embedding.T)):
                add_column_to_layer_tabular_data(labels_layer, "PC_" + str(i), embedding[:, i])

        from ._utilities import show_table
        show_table(self.viewer, labels_layer)

        print("Dimensionality reduction finished")


def umap(reg_props, n_neigh, n_components, standardize):
    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=133, n_components=n_components, n_neighbors=n_neigh)

    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaled_regionprops = StandardScaler().fit_transform(reg_props)
        return reducer.fit_transform(scaled_regionprops)
    else:
        return reducer.fit_transform(reg_props)


def tsne(reg_props, perplexity, n_components, standardize):
    from sklearn.manifold import TSNE

    reducer = TSNE(perplexity=perplexity, n_components=n_components, learning_rate='auto', init='pca', random_state=42)

    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaled_regionprops = StandardScaler().fit_transform(reg_props)
        return reducer.fit_transform(scaled_regionprops)
    else:
        return reducer.fit_transform(reg_props)
        
def pca(reg_props, explained_variance_threshold, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    if n_components == 0 or n_components > len(reg_props.columns):
        pca_object = PCA()
    else:
        pca_object = PCA(n_components=n_components) 

    scaled_regionprops = StandardScaler().fit_transform(reg_props)
    pca_transformed_props = pca_object.fit_transform(scaled_regionprops)

    if n_components == 0:
        explained_variance = pca_object.explained_variance_ratio_
        cumulative_expl_var = [sum(explained_variance[:i+1]) for i in range(len(explained_variance))]
        for i,j in enumerate(cumulative_expl_var):
            if j >= explained_variance_threshold/100:
                pca_cum_var_idx = i
                break
        return pca_transformed_props.T[:pca_cum_var_idx+1].T
    else:
        return pca_transformed_props

    


    
    
