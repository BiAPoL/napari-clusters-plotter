from enum import Enum
import pandas as pd
import warnings
from napari.layers import Labels
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive, restore_defaults
from napari_tools_menu import register_dock_widget
from magicgui.widgets import create_widget
from functools import partial

DEFAULTS = {
    "kmeans_nr_clusters": 2,
    "kmeans_nr_iterations": 3000,
    "normalization": False,
    "hdbscan_min_clusters_size": 5,
    "hdbscan_settings_container_min_nr": 5,
}


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
                                                      options={"choices": [e.value for e in self.Options]})

        self.clust_method_container.layout().addWidget(self.clust_method_choice_list.native)

        # clustering options for KMeans
        # selection of number of clusters
        self.kmeans_settings_container_nr = QWidget()
        self.kmeans_settings_container_nr.setLayout(QHBoxLayout())
        self.kmeans_settings_container_nr.layout().addWidget(QLabel("Number of Clusters"))
        self.kmeans_nr_clusters = create_widget(widget_type="SpinBox",
                                                name="kmeans_nr_clusters",
                                                value=DEFAULTS["kmeans_nr_clusters"],
                                                options={"min": 2, "step": 1})

        self.kmeans_settings_container_nr.layout().addWidget(self.kmeans_nr_clusters.native)
        self.kmeans_settings_container_nr.setVisible(False)

        # selection of number of iterations
        self.kmeans_settings_container_iter = QWidget()
        self.kmeans_settings_container_iter.setLayout(QHBoxLayout())
        self.kmeans_settings_container_iter.layout().addWidget(QLabel("Number of Iterations"))
        self.kmeans_nr_iterations = create_widget(widget_type="SpinBox",
                                                  name="kmeans_nr_iter",
                                                  value=DEFAULTS["kmeans_nr_iterations"],
                                                  options={"min": 1, "max": 10000})

        self.kmeans_settings_container_iter.layout().addWidget(self.kmeans_nr_iterations.native)
        self.kmeans_settings_container_iter.setVisible(False)

        # checkbox whether data should be normalized
        self.clustering_settings_container_scaler = QWidget()
        self.clustering_settings_container_scaler.setLayout(QHBoxLayout())
        self.normalization = create_widget(widget_type="CheckBox", name="Standardize Features",
                                           value=DEFAULTS["normalization"])

        self.clustering_settings_container_scaler.layout().addWidget(self.normalization.native)
        self.clustering_settings_container_scaler.setVisible(False)

        # Clustering options for HDBSCAN
        # selection of the minimum size of clusters
        self.hdbscan_settings_container_size = QWidget()
        self.hdbscan_settings_container_size.setLayout(QHBoxLayout())
        self.hdbscan_settings_container_size.layout().addWidget(QLabel("Minimum size of clusters"))
        self.hdbscan_min_clusters_size = create_widget(widget_type="SpinBox",
                                                       name="hdbscan_min_clusters_size",
                                                       value=DEFAULTS["hdbscan_min_clusters_size"],
                                                       options={"min": 2, "step": 1})

        self.hdbscan_settings_container_size.layout().addWidget(self.hdbscan_min_clusters_size.native)
        self.hdbscan_settings_container_size.setVisible(False)

        # selection of the minimum number of samples in a neighborhood for a point to be considered as a core point
        self.hdbscan_settings_container_min_nr = QWidget()
        self.hdbscan_settings_container_min_nr.setLayout(QHBoxLayout())
        self.hdbscan_settings_container_min_nr.layout().addWidget(QLabel("Minimum number of samples"))
        # hdbscan_min_nr_samples defaults to the min_cluster_size
        self.hdbscan_min_nr_samples = create_widget(widget_type="SpinBox",
                                                    name="hdbscan_min_nr_samples",
                                                    value=self.hdbscan_min_clusters_size.value,
                                                    options={"min": 1, "step": 1})

        self.hdbscan_settings_container_min_nr.layout().addWidget(self.hdbscan_min_nr_samples.native)
        self.hdbscan_settings_container_min_nr.setVisible(False)

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
        self.layout().addWidget(update_container)
        self.layout().addWidget(self.clust_method_container)
        self.layout().addWidget(self.kmeans_settings_container_nr)
        self.layout().addWidget(self.kmeans_settings_container_iter)
        self.layout().addWidget(self.hdbscan_settings_container_size)
        self.layout().addWidget(self.hdbscan_settings_container_min_nr)
        self.layout().addWidget(self.clustering_settings_container_scaler)
        self.layout().addWidget(defaults_container)
        self.layout().addWidget(run_container)
        self.layout().setSpacing(0)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some measurements!")
                return

            if self.clust_method_choice_list.current_choice == "":
                warnings.warn("Please select a clustering method!")
                return

            self.run(
                self.labels_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                self.clust_method_choice_list.current_choice,
                self.kmeans_nr_clusters.value,
                self.kmeans_nr_iterations.value,
                self.normalization.value,
                self.hdbscan_min_clusters_size.value,
                self.hdbscan_min_nr_samples.value
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
        self.clust_method_choice_list.changed.connect(self.change_clustering_options_visibility)

    def change_clustering_options_visibility(self):
        widgets_inactive(self.kmeans_settings_container_nr, self.kmeans_settings_container_iter,
                         active=self.clust_method_choice_list.current_choice == self.Options.KMEANS.value)
        widgets_inactive(self.hdbscan_settings_container_size, self.hdbscan_settings_container_min_nr,
                         active=self.clust_method_choice_list.current_choice == self.Options.HDBSCAN.value)
        widgets_inactive(self.clustering_settings_container_scaler,
                         active=(self.clust_method_choice_list.current_choice == self.Options.KMEANS.value or
                                 self.clust_method_choice_list.current_choice == self.Options.HDBSCAN.value))

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, selected_method, num_clusters, num_iterations, standardize,
            min_cluster_size, min_nr_samples):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))
        print("Selected clustering method: " + str(selected_method))

        # turn properties from layer into a pandas dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        selected_properties = reg_props[selected_measurements_list]

        # perform clustering
        if selected_method == "KMeans":
            y_pred = kmeans_clustering(standardize, selected_properties, num_clusters, num_iterations)
            print("KMeans predictions finished.")
            # write result back to properties of the labels layer
            properties["KMEANS_CLUSTER_ID_SCALER_" + str(standardize)] = y_pred

        elif selected_method == "HDBSCAN":
            y_pred = hdbscan_clustering(standardize, selected_properties, min_cluster_size, min_nr_samples)
            print("HDBSCAN predictions finished.")
            # write result back to properties of the labels layer
            properties["HDBSCAN_CLUSTER_ID_SCALER_" + str(standardize)] = y_pred
        else:
            warnings.warn("Clustering unsuccessful. Please check again selected options.")
            return

        # show region properties table as a new widget
        from ._utilities import show_table
        show_table(self.viewer, labels_layer)


def kmeans_clustering(standardize, measurements, cluster_number, iterations):
    from sklearn.cluster import KMeans
    print("KMeans predictions started...")

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaled_measurements = StandardScaler().fit_transform(measurements)
        # returning prediction as a list for generating clustering image
        return km.fit_predict(scaled_measurements)

    else:
        return km.fit_predict(measurements)


def hdbscan_clustering(standardize, measurements, min_cluster_size, min_samples):
    import hdbscan
    print("HDBSCAN predictions started...")

    clustering_hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaled_measurements = StandardScaler().fit_transform(measurements)
        clustering_hdbscan.fit(scaled_measurements)
        return clustering_hdbscan.fit_predict(scaled_measurements)

    else:
        return clustering_hdbscan.fit_predict(measurements)
