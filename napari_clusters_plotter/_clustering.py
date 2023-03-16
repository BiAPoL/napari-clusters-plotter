import warnings
from enum import Enum
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
from napari.qt.threading import create_worker
from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from ._Qt_code import (
    algorithm_choice,
    button,
    checkbox,
    float_sbox_containter_and_selection,
    int_sbox_containter_and_selection,
    layer_container_and_selection,
    measurements_container_and_list,
    title,
)
from ._utilities import (
    add_column_to_layer_tabular_data,
    buttons_active,
    catch_NaNs,
    get_layer_tabular_data,
    restore_defaults,
    show_table,
    update_properties_list,
    widgets_active,
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
    "ac_n_clusters": 2,
    "ac_n_neighbors": 2,
    "custom_name": "",
}
ID_NAME = "_CLUSTER_ID"


@register_dock_widget(menu="Measurement post-processing > Clustering (ncp)")
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

        title_container = title("<b>Clustering</b>")

        # widget for the selection of labels layer
        (
            labels_layer_selection_container,
            self.labels_select,
        ) = layer_container_and_selection()

        # widget for the selection of properties to perform clustering
        (
            choose_properties_container,
            self.properties_list,
        ) = measurements_container_and_list()

        # selection of the clustering methods
        self.clust_method_container, self.clust_method_choice_list = algorithm_choice(
            name="Clustering_method",
            value=self.Options.EMPTY.value,
            options={"choices": [e.value for e in self.Options]},
            label="Clustering Method",
        )

        # clustering options for KMeans
        # selection of number of clusters
        (
            self.kmeans_settings_container_nr,
            self.kmeans_nr_clusters,
        ) = int_sbox_containter_and_selection(
            name="kmeans_nr_clusters", value=DEFAULTS["kmeans_nr_clusters"]
        )
        # selection of number of iterations
        (
            self.kmeans_settings_container_iter,
            self.kmeans_nr_iterations,
        ) = int_sbox_containter_and_selection(
            name="kmeans_nr_iter",
            value=DEFAULTS["kmeans_nr_iterations"],
            min=1,
            label="Number of Iterations",
        )

        # clustering options for Gaussian mixture model
        # selection of number of clusters
        (
            self.gmm_settings_container_nr,
            self.gmm_nr_clusters,
        ) = int_sbox_containter_and_selection(
            name="gmm_nr_clusters",
            value=DEFAULTS["gmm_nr_clusters"],
        )

        # clustering options for Mean Shift
        # selection of quantile
        (
            self.ms_settings_container_nr,
            self.ms_quantile,
        ) = float_sbox_containter_and_selection(
            name="ms_quantile",
            value=DEFAULTS["ms_quantile"],
            label="Quantile",
        )

        # number of samples selection
        (
            self.ms_settings_container_samples,
            self.ms_n_samples,
        ) = int_sbox_containter_and_selection(
            name="ms_n_samples",
            value=DEFAULTS["ms_n_samples"],
            label="Number of samples",
        )

        # clustering options for Agglomerative Clustering
        # selection of number of clusters
        (
            self.ac_settings_container_clusters,
            self.ac_n_clusters,
        ) = int_sbox_containter_and_selection(
            name="ac_n_clusters",
            value=DEFAULTS["ac_n_clusters"],
        )

        # selection of number of neighbors
        (
            self.ac_settings_container_neighbors,
            self.ac_n_neighbors,
        ) = int_sbox_containter_and_selection(
            name="ac_n_neighbors",
            value=DEFAULTS["ac_n_neighbors"],
            label="Number of neighbors",
        )

        # checkbox whether data should be standardized
        self.clustering_settings_container_scaler, self.standardization = checkbox(
            name="Standardize Features",
            value=DEFAULTS["standardization"],
        )

        # Clustering options for HDBSCAN
        # selection of the minimum size of clusters
        (
            self.hdbscan_settings_container_size,
            self.hdbscan_min_clusters_size,
        ) = int_sbox_containter_and_selection(
            name="hdbscan_min_clusters_size",
            value=DEFAULTS["hdbscan_min_clusters_size"],
            label="Minimum size of clusters",
            tool_link="https://hdbscan.readthedocs.io/en/latest/parameter_selection.html",
            tool_tip=(
                "The minimum size of clusters; single linkage splits that contain fewer points than this will be\n"
                "considered points falling out of a cluster rather than a cluster splitting into two new clusters."
            ),
        )

        # selection of the minimum number of samples in a neighborhood for a point to be considered as a core point
        (
            self.hdbscan_settings_container_min_nr,
            self.hdbscan_min_nr_samples,
        ) = int_sbox_containter_and_selection(
            name="hdbscan_min_nr_samples",
            value=self.hdbscan_min_clusters_size.value,
            min=1,
            label="Minimum number of samples",
            tool_link="https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples",
            tool_tip=(
                "The number of samples in a neighbourhood for a point to be considered a core\n"
                "point. By default it is equal to the minimum cluster size."
            ),
        )

        # custom result column name field
        self.custom_name_container = QWidget()
        self.custom_name_container.setLayout(QHBoxLayout())
        self.custom_name_container.layout().addWidget(QLabel("Custom Results Name"))
        self.custom_name = QLineEdit()
        self.custom_name_not_editable = QLineEdit()

        self.custom_name_container.layout().addWidget(self.custom_name)
        self.custom_name_container.layout().addWidget(self.custom_name_not_editable)
        self.custom_name.setPlaceholderText("Algorithm_name")
        self.custom_name_not_editable.setPlaceholderText(ID_NAME)
        self.custom_name_not_editable.setReadOnly(True)

        # making buttons
        run_container, self.run_button = button("Run")
        update_container, self.update_button = button("Update Measurements")
        defaults_container, self.defaults_button = button("Restore Defaults")

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
        self.layout().addWidget(self.custom_name_container)
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
                self.custom_name.text(),
            )

        self.run_button.clicked.connect(run_clicked)
        self.update_button.clicked.connect(
            partial(update_properties_list, self, [ID_NAME])
        )
        self.defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(
            partial(update_properties_list, self, [ID_NAME])
        )

        # update axes combo boxes automatically if features of
        # layer are changed
        self.last_connected = None
        self.labels_select.changed.connect(self.activate_property_autoupdate)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widget for the selection of parameters unless specific method is chosen
        self.clust_method_choice_list.changed.connect(
            self.change_clustering_options_visibility
        )

        update_properties_list(self, [ID_NAME])

    def change_clustering_options_visibility(self):
        widgets_active(
            self.kmeans_settings_container_nr,
            self.kmeans_settings_container_iter,
            active=self.clust_method_choice_list.current_choice
            == self.Options.KMEANS.value,
        )
        widgets_active(
            self.hdbscan_settings_container_size,
            self.hdbscan_settings_container_min_nr,
            active=self.clust_method_choice_list.current_choice
            == self.Options.HDBSCAN.value,
        )
        widgets_active(
            self.gmm_settings_container_nr,
            active=self.clust_method_choice_list.current_choice
            == self.Options.GMM.value,
        )
        widgets_active(
            self.ms_settings_container_nr,
            self.ms_settings_container_samples,
            active=self.clust_method_choice_list.current_choice
            == self.Options.MS.value,
        )
        widgets_active(
            self.ac_settings_container_clusters,
            self.ac_settings_container_neighbors,
            active=self.clust_method_choice_list.current_choice
            == self.Options.AC.value,
        )

        widgets_active(
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

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                partial(update_properties_list, self, [ID_NAME])
            )
        self.labels_select.value.events.properties.connect(
            partial(update_properties_list, self, [ID_NAME])
        )
        self.last_connected = self.labels_select.value

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
        custom_name,
        show=True,
    ):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))
        print("Selected clustering method: " + str(selected_method))

        features = get_layer_tabular_data(labels_layer)

        # only select the columns the user requested
        selected_properties = features[selected_measurements_list]

        def activate_buttons(active=True):
            """Utility function to enable/disable all the buttons if an error/exception happens in a secondary thread or
            the computation has finished successfully."""

            buttons_active(
                self.run_button, self.defaults_button, self.update_button, active=active
            )

        # disable all the buttons while the computation is happening
        activate_buttons(False)

        # from a secondary thread a tuple (str, np.ndarray) is returned, where str is the name of the clustering method
        def result_of_clustering(returned):
            activate_buttons()

            # write result back to features/properties of the labels layer
            if custom_name == DEFAULTS["custom_name"]:
                result_column_name = returned[0]
            else:
                result_column_name = custom_name
            print(result_column_name + " predictions finished.")
            add_column_to_layer_tabular_data(
                labels_layer,
                result_column_name + ID_NAME,
                returned[1],
            )
            if show:
                show_table(self.viewer, labels_layer)

        # try statement is added to catch any exceptions/errors and enable all the buttons again if that is the case
        try:
            # perform standard scaling, if selected
            if standardize:
                from sklearn.preprocessing import StandardScaler

                selected_properties = StandardScaler().fit_transform(
                    selected_properties
                )

            # perform clustering
            if selected_method == self.Options.KMEANS.value:
                self.worker = create_worker(
                    kmeans_clustering,
                    selected_properties,
                    cluster_number=num_clusters,
                    iterations=num_iterations,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_clustering)
                self.worker.errored.connect(activate_buttons)
                self.worker.start()
            elif selected_method == self.Options.HDBSCAN.value:
                self.worker = create_worker(
                    hdbscan_clustering,
                    selected_properties,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_nr_samples,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_clustering)
                self.worker.errored.connect(activate_buttons)
                self.worker.start()
            elif selected_method == self.Options.GMM.value:
                self.worker = create_worker(
                    gaussian_mixture_model,
                    selected_properties,
                    cluster_number=gmm_num_cluster,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_clustering)
                self.worker.errored.connect(activate_buttons)
                self.worker.start()
            elif selected_method == self.Options.MS.value:
                self.worker = create_worker(
                    mean_shift,
                    selected_properties,
                    quantile=ms_quantile,
                    n_samples=ms_n_samples,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_clustering)
                self.worker.errored.connect(activate_buttons)
                self.worker.start()
            elif selected_method == self.Options.AC.value:
                self.worker = create_worker(
                    agglomerative_clustering,
                    selected_properties,
                    cluster_number=ac_n_clusters,
                    n_neighbors=ac_n_neighbors,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_clustering)
                self.worker.errored.connect(activate_buttons)
                self.worker.start()
            else:
                warnings.warn(
                    "Clustering unsuccessful. Please check selected options again."
                )
                return

        except Exception:
            # make buttons active again even if an exception occurred during execution of the code above and not
            # in a secondary thread
            activate_buttons()


@catch_NaNs
def mean_shift(
    reg_props: pd.DataFrame, quantile: float = 0.2, n_samples: int = 50
) -> Tuple[str, np.ndarray]:
    from sklearn.cluster import MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(reg_props, quantile=quantile, n_samples=n_samples)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return "MS", ms.fit_predict(reg_props)


@catch_NaNs
def gaussian_mixture_model(
    reg_props: pd.DataFrame, cluster_number: int
) -> Tuple[str, np.ndarray]:
    from sklearn import mixture

    # fit a Gaussian Mixture Model
    gmm = mixture.GaussianMixture(n_components=cluster_number, covariance_type="full")

    return "GMM", gmm.fit_predict(reg_props)


@catch_NaNs
def kmeans_clustering(
    reg_props: pd.DataFrame, cluster_number: int, iterations: int
) -> Tuple[str, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    return "KMEANS", km.fit_predict(reg_props)


@catch_NaNs
def agglomerative_clustering(
    reg_props: pd.DataFrame, cluster_number: int, n_neighbors: int
) -> Tuple[str, np.ndarray]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    # source: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        reg_props, n_neighbors=n_neighbors, include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    ac = AgglomerativeClustering(
        n_clusters=cluster_number, linkage="ward", connectivity=connectivity
    )

    return "AC", ac.fit_predict(reg_props)


@catch_NaNs
def hdbscan_clustering(
    reg_props: pd.DataFrame, min_cluster_size: int, min_samples: int
) -> Tuple[str, np.ndarray]:
    import hdbscan

    clustering_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )

    return "HDBSCAN", clustering_hdbscan.fit_predict(reg_props)
