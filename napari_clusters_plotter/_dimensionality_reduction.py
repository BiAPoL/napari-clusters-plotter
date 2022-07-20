import warnings
from functools import partial
from typing import Tuple
from enum import Enum
import numpy as np
import pandas as pd
from napari.qt.threading import create_worker
from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import QVBoxLayout, QWidget


from ._plotter import POINTER
from ._clustering import ID_NAME
from ._utilities import (
    add_column_to_layer_tabular_data,
    catch_NaNs,
    get_layer_tabular_data,
    restore_defaults,
    set_features,
    show_table,
    widgets_inactive,
    update_properties_list,
)

from ._Qt_code import (
    measurements_container_and_list,
    labels_container_and_selection,
    int_sbox_containter_and_selection,
    float_sbox_containter_and_selection,
    title,
    button,
    checkbox,
    algorithm_choice,
)

# Remove when the problem is fixed from sklearn side
warnings.filterwarnings(action="ignore", category=FutureWarning, module="sklearn")

DEFAULTS = {
    "n_neighbors": 15,
    "perplexity": 30,
    "standardization": True,
    "pca_components": 0,
    "explained_variance": 95.0,
}
EXCLUDE = [ID_NAME,POINTER,"UMAP","t-SNE"]

@register_dock_widget(menu="Measurement > Dimensionality reduction (ncp)")
class DimensionalityReductionWidget(QWidget):
    class Options(Enum):
        EMPTY = ""
        UMAP = "UMAP"
        TSNE = "t-SNE"
        PCA = "PCA"

    def __init__(self, napari_viewer):
        super().__init__()

        self.worker = None
        self.viewer = napari_viewer

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = title("<b>Dimensionality reduction</b>")

        # widget for the selection of labels layer
        labels_layer_selection_container, self.labels_select= labels_container_and_selection()

        # select properties of which to produce a dimensionality reduced version
        choose_properties_container,self.properties_list = measurements_container_and_list()

        # selection of dimension reduction algorithm
        algorithm_container,self.algorithm_choice_list = algorithm_choice(
            name="Clustering_method",
            value=self.Options.EMPTY.value,
            options={"choices": [e.value for e in self.Options]},
            label="Clustering Method",
        )

        # selection of n_neighbors - The size of local neighborhood (in terms of number of neighboring sample points)
        # used for manifold approximation. Larger values result in more global views of the manifold, while smaller
        # values result in more local data being preserved.
        self.n_neighbors_container,self.n_neighbors = int_sbox_containter_and_selection(
            name="n_neighbors",
            value=DEFAULTS["n_neighbors"],
            label ="Number of neighbors",
            tool_link='https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors',
            tool_tip=(
                "The size of local neighborhood (in terms of number of neighboring sample points) used for manifold\n"
                "approximation. Larger values result in more global views of the manifold, while smaller values\n"
                "result in more local data being preserved. In general, it should be in the range 2 to 100."
            )
        )
        
        # selection of the level of perplexity. Higher values should be chosen when handling large datasets
        self.perplexity_container,self.perplexity = int_sbox_containter_and_selection(
            name="perplexity",
            value=DEFAULTS["perplexity"],
            label="Perplexity",
            min=1,
            tool_link='https://distill.pub/2016/misread-tsne/',
            tool_tip=(
                "The perplexity is related to the number of nearest neighbors that is used in other manifold learning\n"
                "algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and\n"
                "50. Different values can result in significantly different results."
            )
        )

        # selection of the number of components to keep after PCA transformation,
        # values above 0 will override explained variance option
        self.pca_components_container,self.pca_components = int_sbox_containter_and_selection(
            name="pca_components",
            value=DEFAULTS["pca_components"],
            min=0,
            label ="Number of Components",
            tool_link='https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
            tool_tip=(
                "The number of components sets the number of principal components to be included after the transformation.\n"
                "When set to 0 the number of components that are selected is determined by the explained variance\n"
                "threshold."
            )
        )

        # Minimum percentage of variance explained by kept PCA components,
        # will not be used if pca_components > 0
        self.explained_variance_container,self.explained_variance = float_sbox_containter_and_selection(
            name="explained_variance",
            value=DEFAULTS["explained_variance"],
            min=1, max=100, step=1,
            label="Explained Variance Threshold",
            tool_link='https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
            tool_tip=(
                "The explained variance threshold sets the amount of variance in the dataset that can "
                "minimally be\n represented by the principal components. The closer the threshold is to"
                " 100% ,the more the variance in\nthe dataset can be accounted for by the chosen "
                "principal components (and the less dimensionality\nreduction will be perfomed as a result)."
            )
        )

        # checkbox whether data should be standardized
        self.settings_container_scaler,self.standardization= checkbox(
            name="Standardize Features",
            value=DEFAULTS["standardization"],
        )

        # making buttons
        run_container,run_button = button("Run")
        update_container,update_button = button("Update Measurements")
        defaults_container,defaults_button = button("Restore Defaults")

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some measurements!")
                return

            if self.algorithm_choice_list.current_choice == self.Options.EMPTY.value:
                warnings.warn("Please select dimensionality reduction algorithm.")
                return

            self.run(
                self.viewer,
                self.labels_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                self.n_neighbors.value,
                self.perplexity.value,
                self.algorithm_choice_list.current_choice,
                self.standardization.value,
                self.explained_variance.value,
                self.pca_components.value,
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(partial(update_properties_list,self,EXCLUDE))
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(partial(update_properties_list,self,EXCLUDE))

        self.last_connected = None
        self.labels_select.changed.connect(self.activate_property_autoupdate)

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
        self.layout().addWidget(run_container)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.algorithm_choice_list.changed.connect(
            self.change_settings_visibility
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_settings_visibility(self):
        widgets_inactive(
            self.n_neighbors_container,
            active=self.algorithm_choice_list.current_choice 
            == self.Options.UMAP.value,
        )
        widgets_inactive(
            self.settings_container_scaler,
            active=(
                self.algorithm_choice_list.current_choice 
                == self.Options.UMAP.value
                or self.algorithm_choice_list.current_choice 
                == self.Options.TSNE.value
            ),
        )
        widgets_inactive(
            self.perplexity_container,
            active=self.algorithm_choice_list.current_choice 
            == self.Options.TSNE.value,
        )
        widgets_inactive(
            self.pca_components_container,
            active=self.algorithm_choice_list.current_choice
            == self.Options.PCA.value,
        )
        widgets_inactive(
            self.explained_variance_container,
            active=self.algorithm_choice_list.current_choice
            == self.Options.PCA.value,
        )
        

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                partial(update_properties_list,self,EXCLUDE)
            )
        self.labels_select.value.events.properties.connect(partial(update_properties_list,self,EXCLUDE))
        self.last_connected = self.labels_select.value

    # this function runs after the run button is clicked
    def run(
        self,
        viewer,
        labels_layer,
        selected_measurements_list,
        n_neighbours,
        perplexity,
        selected_algorithm,
        standardize,
        explained_variance,
        pca_components,
        n_components=2,  # dimension of the embedded space. For now 2 by default, since only 2D plotting is supported
    ):

        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        features = get_layer_tabular_data(labels_layer)

        # only select the columns the user requested
        properties_to_reduce = features[selected_measurements_list]

        # perform standard scaling, if selected
        if standardize:
            from sklearn.preprocessing import StandardScaler

            properties_to_reduce = StandardScaler().fit_transform(properties_to_reduce)

        # from a secondary thread a tuple[str, np.ndarray] is returned, where result[0] is the name of algorithm
        def return_func_dim_reduction(result):

            if result[0] == "PCA":
                # check if principal components are already present
                # and remove them by overwriting the features
                tabular_data = get_layer_tabular_data(labels_layer)
                dropkeys = [
                    column for column in tabular_data.keys() if column.startswith("PC_")
                ]
                df_principal_components_removed = tabular_data.drop(dropkeys, axis=1)
                set_features(labels_layer, df_principal_components_removed)

                # write result back to properties
                for i in range(0, len(result[1].T)):
                    add_column_to_layer_tabular_data(
                        labels_layer, "PC_" + str(i), result[1][:, i]
                    )

            elif result[0] == "UMAP" or result[0] == "t-SNE":
                # write result back to properties
                for i in range(0, n_components):
                    add_column_to_layer_tabular_data(
                        labels_layer, result[0] + "_" + str(i), result[1][:, i]
                    )

            else:
                "Dimensionality reduction not successful. Please try again"
                return

            show_table(viewer, labels_layer)
            print("Dimensionality reduction finished")

        # depending on the selected dim reduction algorithm start a secondary thread
        if selected_algorithm == self.Options.UMAP.value:
            self.worker = create_worker(
                umap,
                properties_to_reduce,
                n_neigh=n_neighbours,
                n_components=n_components,
                _progress=True,
            )
            self.worker.returned.connect(return_func_dim_reduction)
            self.worker.start()

        elif selected_algorithm == self.Options.TSNE.value:
            self.worker = create_worker(
                tsne,
                properties_to_reduce,
                perplexity=perplexity,
                n_components=n_components,
                _progress=True,
            )
            self.worker.returned.connect(return_func_dim_reduction)
            self.worker.start()

        elif selected_algorithm == self.Options.PCA.value:
            self.worker = create_worker(
                pca,
                properties_to_reduce,
                explained_variance_threshold=explained_variance,
                n_components=pca_components,
                _progress=True,
            )
            self.worker.returned.connect(return_func_dim_reduction)
            self.worker.start()


@catch_NaNs
def umap(
    reg_props: pd.DataFrame, n_neigh: int, n_components: int
) -> Tuple[str, np.ndarray]:
    import umap.umap_ as umap

    reducer = umap.UMAP(
        random_state=133,
        n_components=n_components,
        n_neighbors=n_neigh,
        verbose=True,
        tqdm_kwds={"desc": "Dimensionality reduction progress"},
    )
    return "UMAP", reducer.fit_transform(reg_props)


@catch_NaNs
def tsne(
    reg_props: pd.DataFrame, perplexity: float, n_components: int
) -> Tuple[str, np.ndarray]:
    from sklearn.manifold import TSNE

    reducer = TSNE(
        perplexity=perplexity,
        n_components=n_components,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    return "t-SNE", reducer.fit_transform(reg_props)


@catch_NaNs
def pca(
    reg_props: pd.DataFrame, explained_variance_threshold: float, n_components: int
) -> Tuple[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if n_components == 0 or n_components > len(reg_props.columns):
        pca_object = PCA()
    else:
        pca_object = PCA(n_components=n_components)

    scaled_regionprops = StandardScaler().fit_transform(reg_props)
    pca_transformed_props = pca_object.fit_transform(scaled_regionprops)

    if n_components == 0:
        explained_variance = pca_object.explained_variance_ratio_
        cumulative_expl_var = [
            sum(explained_variance[: i + 1]) for i in range(len(explained_variance))
        ]
        for i, j in enumerate(cumulative_expl_var):
            if j >= explained_variance_threshold / 100:
                pca_cum_var_idx = i
                break
        return "PCA", pca_transformed_props.T[: pca_cum_var_idx + 1].T
    else:
        return "PCA", pca_transformed_props
