import warnings
from enum import Enum
from functools import partial

from magicgui.widgets import create_widget
from napari.layers import Labels
from napari.qt.threading import create_worker
from napari_tools_menu import register_dock_widget
from qtpy.QtCore import QRect
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._utilities import (
    add_column_to_layer_tabular_data,
    get_layer_tabular_data,
    restore_defaults,
    show_table,
    widgets_inactive,
)

DEFAULTS = {
    "kmeans_nr_clusters": 2,
    "kmeans_nr_iterations": 300,
    "standardization": False,
    "hdbscan_min_clusters_size": 5,
    "hdbscan_min_nr_samples": 5,
    "gmm_nr_clusters": 2,
    "ms_quantile": 0.2,
    "ms_n_samples": 50,
    "ac_nr_clusters": 2,
    "ac_nr_neighbors": 2,
}


@register_dock_widget(menu="Measurement > Clustering (ncp)")
class ClusteringWidget(QWidget):
    class Options(Enum):
        EMPTY = ""
        KMEANS = "KMeans"
        HDBSCAN = "HDBSCAN"
        GMM = "Gaussian Mixture Model (GMM)"
        MS = "Mean Shift (MS)"
        AC = "Agglomerative Clustering (AC)"

    def __init__(self, napari_viewer):
        super().__init__()
        self.worker = None
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
        self.clust_method_choice_list = create_widget(
            widget_type="ComboBox",
            name="Clustering_method",
            value=self.Options.EMPTY.value,
            options={"choices": [e.value for e in self.Options]},
        )

        self.clust_method_container.layout().addWidget(
            self.clust_method_choice_list.native
        )

        # clustering options for KMeans
        # selection of number of clusters
        self.kmeans_settings_container_nr = QWidget()
        self.kmeans_settings_container_nr.setLayout(QHBoxLayout())
        self.kmeans_settings_container_nr.layout().addWidget(
            QLabel("Number of Clusters")
        )
        self.kmeans_nr_clusters = create_widget(
            widget_type="SpinBox",
            name="kmeans_nr_clusters",
            value=DEFAULTS["kmeans_nr_clusters"],
            options={"min": 2, "step": 1},
        )

        self.kmeans_settings_container_nr.layout().addWidget(
            self.kmeans_nr_clusters.native
        )
        self.kmeans_settings_container_nr.setVisible(False)

        # selection of number of iterations
        self.kmeans_settings_container_iter = QWidget()
        self.kmeans_settings_container_iter.setLayout(QHBoxLayout())
        self.kmeans_settings_container_iter.layout().addWidget(
            QLabel("Number of Iterations")
        )
        self.kmeans_nr_iterations = create_widget(
            widget_type="SpinBox",
            name="kmeans_nr_iter",
            value=DEFAULTS["kmeans_nr_iterations"],
            options={"min": 1, "max": 10000},
        )

        self.kmeans_settings_container_iter.layout().addWidget(
            self.kmeans_nr_iterations.native
        )
        self.kmeans_settings_container_iter.setVisible(False)

        # clustering options for Gaussian mixture model
        # selection of number of clusters
        self.gmm_settings_container_nr = QWidget()
        self.gmm_settings_container_nr.setLayout(QHBoxLayout())
        self.gmm_settings_container_nr.layout().addWidget(QLabel("Number of Clusters"))
        self.gmm_nr_clusters = create_widget(
            widget_type="SpinBox",
            name="gmm_nr_clusters",
            value=DEFAULTS["gmm_nr_clusters"],
            options={"min": 2, "step": 1},
        )

        self.gmm_settings_container_nr.layout().addWidget(self.gmm_nr_clusters.native)
        self.gmm_settings_container_nr.setVisible(False)

        # clustering options for Mean Shift
        # selection of quantile
        self.ms_settings_container_nr = QWidget()
        self.ms_settings_container_nr.setLayout(QHBoxLayout())
        self.ms_settings_container_nr.layout().addWidget(QLabel("Quantile"))
        self.ms_quantile = create_widget(
            widget_type="FloatSpinBox",
            name="ms_quantile",
            value=DEFAULTS["ms_quantile"],
            options={"min": 0, "step": 0.1, "max": 1},
        )

        self.ms_settings_container_nr.layout().addWidget(self.ms_quantile.native)
        self.ms_settings_container_nr.setVisible(False)

        # selection of number of samples
        self.ms_settings_container_samples = QWidget()
        self.ms_settings_container_samples.setLayout(QHBoxLayout())
        self.ms_settings_container_samples.layout().addWidget(
            QLabel("Number of samples")
        )
        self.ms_n_samples = create_widget(
            widget_type="SpinBox",
            name="ms_n_samples",
            value=DEFAULTS["ms_n_samples"],
            options={"min": 2, "step": 1},
        )

        self.ms_settings_container_samples.layout().addWidget(self.ms_n_samples.native)
        self.ms_settings_container_samples.setVisible(False)

        #
        # clustering options for Agglomerative Clustering
        # selection of number of clusters
        self.ac_settings_container_clusters = QWidget()
        self.ac_settings_container_clusters.setLayout(QHBoxLayout())
        self.ac_settings_container_clusters.layout().addWidget(
            QLabel("Number of clusters")
        )
        self.ac_n_clusters = create_widget(
            widget_type="SpinBox",
            name="ac_nr_clusters",
            value=DEFAULTS["ac_nr_clusters"],
            options={"min": 2, "step": 1},
        )

        self.ac_settings_container_clusters.layout().addWidget(
            self.ac_n_clusters.native
        )
        self.ac_settings_container_clusters.setVisible(False)

        # selection of number of clusters
        self.ac_settings_container_neighbors = QWidget()
        self.ac_settings_container_neighbors.setLayout(QHBoxLayout())
        self.ac_settings_container_neighbors.layout().addWidget(
            QLabel("Number of neighbors")
        )
        self.ac_n_neighbors = create_widget(
            widget_type="SpinBox",
            name="ac_nr_neighbors",
            value=DEFAULTS["ac_nr_neighbors"],
            options={"min": 2, "step": 1},
        )

        self.ac_settings_container_neighbors.layout().addWidget(
            self.ac_n_neighbors.native
        )
        self.ac_settings_container_neighbors.setVisible(False)

        # checkbox whether data should be standardized
        self.clustering_settings_container_scaler = QWidget()
        self.clustering_settings_container_scaler.setLayout(QHBoxLayout())
        self.standardization = create_widget(
            widget_type="CheckBox",
            name="Standardize Features",
            value=DEFAULTS["standardization"],
        )

        self.clustering_settings_container_scaler.layout().addWidget(
            self.standardization.native
        )
        self.clustering_settings_container_scaler.setVisible(False)

        # Clustering options for HDBSCAN
        # selection of the minimum size of clusters
        self.hdbscan_settings_container_size = QWidget()
        self.hdbscan_settings_container_size.setLayout(QHBoxLayout())
        self.hdbscan_settings_container_size.layout().addWidget(
            QLabel("Minimum size of clusters")
        )
        self.hdbscan_settings_container_size.layout().addStretch()
        self.hdbscan_min_clusters_size = create_widget(
            widget_type="SpinBox",
            name="hdbscan_min_clusters_size",
            value=DEFAULTS["hdbscan_min_clusters_size"],
            options={"min": 2, "step": 1},
        )

        help_min_clusters_size = QLabel()
        help_min_clusters_size.setOpenExternalLinks(True)
        help_min_clusters_size.setText(
            '<a href="https://hdbscan.readthedocs.io/en/latest/parameter_selection.html" '
            'style="text-decoration:none; color:white"><b>?</b></a>'
        )

        help_min_clusters_size.setToolTip(
            "The minimum size of clusters; single linkage splits that contain fewer points than this will be "
            "considered points falling out of a cluster rather than a cluster splitting into two new clusters. "
            " Click on question mark to read more."
        )

        self.hdbscan_min_clusters_size.native.setMaximumWidth(70)
        self.hdbscan_settings_container_size.layout().addWidget(
            self.hdbscan_min_clusters_size.native
        )
        self.hdbscan_settings_container_size.layout().addWidget(help_min_clusters_size)
        self.hdbscan_settings_container_size.setVisible(False)

        # selection of the minimum number of samples in a neighborhood for a point to be considered as a core point
        self.hdbscan_settings_container_min_nr = QWidget()
        self.hdbscan_settings_container_min_nr.setLayout(QHBoxLayout())
        self.hdbscan_settings_container_min_nr.layout().addWidget(
            QLabel("Minimum number of samples")
        )
        self.hdbscan_settings_container_min_nr.layout().addStretch()
        self.hdbscan_min_nr_samples = create_widget(
            widget_type="SpinBox",
            name="hdbscan_min_nr_samples",
            value=self.hdbscan_min_clusters_size.value,
            options={"min": 1, "step": 1},
        )
        help_min_nr_samples = QLabel()
        help_min_nr_samples.setOpenExternalLinks(True)
        help_min_nr_samples.setText(
            '<a href="https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples" '
            'style="text-decoration:none; color:white"><b>?</b></a>'
        )

        help_min_nr_samples.setToolTip(
            "The number of samples in a neighbourhood for a point to be considered a core "
            "point. By default it is equal to the minimum cluster size. Click on the "
            "question mark to read more."
        )

        self.hdbscan_min_nr_samples.native.setMaximumWidth(70)
        self.hdbscan_settings_container_min_nr.layout().addWidget(
            self.hdbscan_min_nr_samples.native
        )
        self.hdbscan_settings_container_min_nr.layout().addWidget(help_min_nr_samples)
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
        self.layout().addWidget(self.gmm_settings_container_nr)
        self.layout().addWidget(self.ms_settings_container_nr)
        self.layout().addWidget(self.ms_settings_container_samples)
        self.layout().addWidget(self.ac_settings_container_clusters)
        self.layout().addWidget(self.ac_settings_container_neighbors)
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
                self.standardization.value,
                self.hdbscan_min_clusters_size.value,
                self.hdbscan_min_nr_samples.value,
                self.gmm_nr_clusters.value,
                self.ms_quantile.value,
                self.ms_n_samples.value,
                self.ac_n_clusters.value,
                self.ac_n_neighbors.value,
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
        self.clust_method_choice_list.changed.connect(
            self.change_clustering_options_visibility
        )

    def change_clustering_options_visibility(self):
        widgets_inactive(
            self.kmeans_settings_container_nr,
            self.kmeans_settings_container_iter,
            active=self.clust_method_choice_list.current_choice
            == self.Options.KMEANS.value,
        )
        widgets_inactive(
            self.hdbscan_settings_container_size,
            self.hdbscan_settings_container_min_nr,
            active=self.clust_method_choice_list.current_choice
            == self.Options.HDBSCAN.value,
        )
        widgets_inactive(
            self.gmm_settings_container_nr,
            active=self.clust_method_choice_list.current_choice
            == self.Options.GMM.value,
        )
        widgets_inactive(
            self.ms_settings_container_nr,
            self.ms_settings_container_samples,
            active=self.clust_method_choice_list.current_choice
            == self.Options.MS.value,
        )
        widgets_inactive(
            self.ac_settings_container_clusters,
            self.ac_settings_container_neighbors,
            active=self.clust_method_choice_list.current_choice
            == self.Options.AC.value,
        )

        widgets_inactive(
            self.clustering_settings_container_scaler,
            active=(
                self.clust_method_choice_list.current_choice
                == self.Options.KMEANS.value
                or self.clust_method_choice_list.current_choice
                == self.Options.HDBSCAN.value
                or self.clust_method_choice_list.current_choice
                == self.Options.GMM.value
                or self.clust_method_choice_list.current_choice == self.Options.MS.value
                or self.clust_method_choice_list.current_choice == self.Options.AC.value
            ),
        )

    def update_properties_list(self):
        selected_layer = self.labels_select.value

        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if "label" in p or "CLUSTER_ID" in p or "index" in p:
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
    def run(
        self,
        labels_layer,
        selected_measurements_list,
        selected_method,
        num_clusters,
        num_iterations,
        standardize,
        min_cluster_size,
        min_nr_samples,
        gmm_num_cluster,
        ms_quantile,
        ms_n_samples,
        ac_n_clusters,
        ac_n_neighbors,
    ):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))
        print("Selected clustering method: " + str(selected_method))

        features = get_layer_tabular_data(labels_layer)

        # only select the columns the user requested
        selected_properties = features[selected_measurements_list]

        # from a secondary thread a tuple is returned, where the first item (returned[0]) is the name of
        # the clustering method, and the second one is predictions (returned[1])
        def result_of_clustering(returned):
            print(returned[0] + " predictions finished.")
            # write result back to features/properties of the labels layer
            add_column_to_layer_tabular_data(
                labels_layer,
                returned[1] + "_CLUSTER_ID_SCALER_" + str(standardize),
                returned[1],
            )
            show_table(self.viewer, labels_layer)

        # perform standard scaling, if selected
        if standardize:
            from sklearn.preprocessing import StandardScaler

            selected_properties = StandardScaler().fit_transform(selected_properties)

        # perform clustering
        if selected_method == "KMeans":
            self.worker = create_worker(
                kmeans_clustering,
                measurements=selected_properties,
                cluster_number=num_clusters,
                iterations=num_iterations,
                _progress=True,
            )
            self.worker.returned.connect(result_of_clustering)
            self.worker.start()
        elif selected_method == "HDBSCAN":
            self.worker = create_worker(
                hdbscan_clustering,
                measurements=selected_properties,
                min_cluster_size=min_cluster_size,
                min_samples=min_nr_samples,
                _progress=True,
            )
            self.worker.returned.connect(result_of_clustering)
            self.worker.start()
        elif selected_method == "Gaussian Mixture Model (GMM)":
            self.worker = create_worker(
                gaussian_mixture_model,
                measurements=selected_properties,
                cluster_number=gmm_num_cluster,
                _progress=True,
            )
            self.worker.returned.connect(result_of_clustering)
            self.worker.start()
        elif selected_method == "Mean Shift (MS)":
            self.worker = create_worker(
                gaussian_mixture_model,
                measurements=selected_properties,
                quantile=ms_quantile,
                n_samples=ms_n_samples,
                _progress=True,
            )
            self.worker.returned.connect(result_of_clustering)
            self.worker.start()
        elif selected_method == "Agglomerative Clustering (AC)":
            self.worker = create_worker(
                gaussian_mixture_model,
                measurements=selected_properties,
                cluster_number=ac_n_clusters,
                n_neighbors=ac_n_neighbors,
                _progress=True,
            )
            self.worker.returned.connect(result_of_clustering)
            self.worker.start()
        else:
            warnings.warn(
                "Clustering unsuccessful. Please check selected options again."
            )
            return


def mean_shift(measurements, quantile=0.2, n_samples=50):
    from sklearn.cluster import MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(measurements, quantile=quantile, n_samples=n_samples)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return "MS", ms.fit_predict(measurements)


def gaussian_mixture_model(measurements, cluster_number):
    from sklearn import mixture

    # fit a Gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components=cluster_number, covariance_type="full")

    return "GMM", gmm.fit_predict(measurements)


def kmeans_clustering(measurements, cluster_number, iterations):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    return "KMEANS", km.fit_predict(measurements)


def agglomerative_clustering(measurements, cluster_number, n_neighbors):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    # source: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        measurements, n_neighbors=n_neighbors, include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    ac = AgglomerativeClustering(
        n_clusters=cluster_number, linkage="ward", connectivity=connectivity
    )

    return "AC", ac.fit_predict(measurements)


def hdbscan_clustering(measurements, min_cluster_size, min_samples):
    import hdbscan

    clustering_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )

    return "HDBSCAN", clustering_hdbscan.fit_predict(measurements)
