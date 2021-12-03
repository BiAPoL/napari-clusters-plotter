from enum import Enum
import pandas as pd
import warnings
from napari.layers import Labels
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive
from ._utilities import restore_defaults
from napari_tools_menu import register_dock_widget
from magicgui.widgets import create_widget
from functools import partial

DEFAULTS = dict(
    kmeans_nr_clusters=2,
    kmeans_nr_iterations=3000,
)


@register_dock_widget(menu="Measurement > Clustering (ncp)")
class ClusteringWidget(QWidget):

    class Options(Enum):
        EMPTY = ""
        KMEANS = "KMeans"
        HDBSCAN = "HDBSCAN"

    def __init__(self, napari_viewer):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.viewer = napari_viewer

        title_container = QWidget()
        title_container.setLayout(QVBoxLayout())
        title_container.layout().addWidget(QLabel("<b>Clustering</b>"))

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # widget for the selection of properties to perform clustering
        choose_properties_container = QWidget()
        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Measurements"))
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))
        choose_properties_container.layout().addWidget(self.properties_list)

        # selection of the clustering methods
        self.clust_method_container = QWidget()
        self.clust_method_container.setLayout(QHBoxLayout())
        self.clust_method_container.layout().addWidget(QLabel("Clustering Method"))
        self.clust_method_choice_list = create_widget(widget_type="ComboBox",
                                                      name="Clustering_method",
                                                      value=self.Options.EMPTY.value,
                                                      options=dict(choices=[e.value for e in self.Options]))

        self.clust_method_container.layout().addWidget(self.clust_method_choice_list.native)

        # clustering options for KMeans
        # selection of number of clusters
        self.kmeans_settings_container_nr = QWidget()
        self.kmeans_settings_container_nr.setLayout(QHBoxLayout())
        self.kmeans_settings_container_nr.layout().addWidget(QLabel("Number of Clusters"))
        self.kmeans_nr_clusters = create_widget(widget_type="SpinBox",
                                                name='kmeans_nr_clusters',
                                                value=DEFAULTS['kmeans_nr_clusters'],
                                                options=dict(min=2, step=1))

        self.kmeans_settings_container_nr.layout().addWidget(self.kmeans_nr_clusters.native)
        self.kmeans_settings_container_nr.setVisible(False)

        # selection of number of iterations
        self.kmeans_settings_container_iter = QWidget()
        self.kmeans_settings_container_iter.setLayout(QHBoxLayout())
        self.kmeans_settings_container_iter.layout().addWidget(QLabel("Number of Iterations"))
        self.kmeans_nr_iterations = create_widget(widget_type="SpinBox",
                                                  name='kmeans_nr_iter',
                                                  value=DEFAULTS['kmeans_nr_iterations'],
                                                  options=dict(min=1, max=10000))

        self.kmeans_settings_container_iter.layout().addWidget(self.kmeans_nr_iterations.native)
        self.kmeans_settings_container_iter.setVisible(False)

        # Clustering options for HDBSCAN
        # Todo

        # Run button
        run_container = QWidget()
        run_container.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_container.layout().addWidget(run_button)

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

        # adding all widgets to the layout
        self.layout().addWidget(title_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(self.clust_method_container)
        self.layout().addWidget(self.kmeans_settings_container_nr)
        self.layout().addWidget(self.kmeans_settings_container_iter)
        self.layout().addWidget(run_container)
        self.layout().addWidget(defaults_container)
        self.layout().addWidget(update_container)
        self.layout().setSpacing(0)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            self.run(
                self.labels_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                self.kmeans_nr_clusters.value,
                self.kmeans_nr_iterations.value
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(self.update_properties_list)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widget for the selection of parameters unless specific method is chosen
        self.clust_method_choice_list.changed.connect(self.change_kmeans_clustering)

    def change_kmeans_clustering(self):
        widgets_inactive(self.kmeans_settings_container_nr, self.kmeans_settings_container_iter,
                         active=self.clust_method_choice_list.current_choice == self.Options.KMEANS.value)

    def update_properties_list(self):
        selected_layer = self.labels_select.value

        if selected_layer is not None:
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                self.properties_list.clear()
                for p in list(properties.keys()):
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    # per default select all measurements that are not "label"
                    if p != "label":
                        item.setSelected(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()
        self.update_properties_list()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, num_clusters, num_iterations):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        # turn properties from layer into a pandas dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        selected_properties = reg_props[selected_measurements_list]

        # perform clustering
        y_pred = kmeans_clustering(selected_properties, num_clusters, num_iterations)
        print('KMeans predictions finished.')

        # write result back to properties of the labels layer
        properties["KMEANS_CLUSTER_ID"] = y_pred

        # show region properties table as a new widget
        from ._utilities import show_table
        show_table(self.viewer, labels_layer)


def kmeans_clustering(measurements, cluster_number, iterations):
    from sklearn.cluster import KMeans
    print('KMeans predictions started...')

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    # returning prediction as a list for generating clustering image
    return km.fit_predict(measurements)

