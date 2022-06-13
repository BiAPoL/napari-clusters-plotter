import warnings
from functools import partial
import pandas as pd
import numpy as np

from magicgui.widgets import create_widget
from napari.layers import Labels
from napari_tools_menu import register_dock_widget
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

from ._utilities import (
    get_layer_tabular_data,
    restore_defaults,
    set_features,
    show_table,
    widgets_inactive,
)
from._plotter import POINTER
DEFAULTS = dict(correlation_threshold=0.95)
NON_DATA_COLUMN_NAMES = ["label", POINTER, "index"]


@register_dock_widget(menu="Measurement > Feature selection (ncp)")
class FeatureSelectionWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Feature Selection</b>"))

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # selection of feature selection method
        method_container = QWidget()
        method_container.setLayout(QHBoxLayout())
        method_container.layout().addWidget(QLabel("Feature Selection Method"))
        self.method_choice_list = QComboBox()
        self.method_choice_list.addItems(["", "Correlation Filter"])
        method_container.layout().addWidget(self.method_choice_list)

        # Threshold of Pearson's correlation at which two features are categorised as correlating
        self.correlation_threshold_container = QWidget()
        self.correlation_threshold_container.setLayout(QHBoxLayout())
        self.correlation_threshold_container.layout().addWidget(
            QLabel("Number of neighbors")
        )
        self.correlation_threshold = create_widget(
            widget_type="FloatSpinBox",
            name="correlation_threshold",
            value=DEFAULTS["correlation_threshold"],
            options=dict(min=0, max=1, step=0.01),
        )

        self.correlation_threshold_container.layout().addWidget(
            self.correlation_threshold.native
        )
        self.correlation_threshold_container.setVisible(False)

        # get all properties to be able to perform feature selection
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

            if self.method_choice_list.currentText() == "":
                warnings.warn("Please select feature selection method.")
                return

            self.run(self.labels_select.value)

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))
        
        self.labels_select.changed.connect(self.update_properties_list)        
        self.last_connected = None
        self.labels_select.changed.connect(self.activate_property_autoupdate)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(method_container)
        self.layout().addWidget(self.correlation_threshold_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(defaults_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(run_widget)


        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.method_choice_list.currentIndexChanged.connect(
            self.change_correlation_threshold
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_correlation_threshold(self):
        widgets_inactive(
            self.correlation_threshold_container,
            active=self.method_choice_list.currentText() == "Correlation Filter",
        )


    def update_properties_list(self):
        selected_layer = self.labels_select.value
        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if (
                        "label" in p
                        or "CLUSTER_ID" in p
                        or "index" in p
                        or POINTER in p
                    ):
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)
    
    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                self.update_properties_list
            )
        self.labels_select.value.events.properties.connect(self.update_properties_list)
        self.last_connected = self.labels_select.value

    # this function runs after the run button is clicked
    def run(self, labels_layer):
        reg_props = get_layer_tabular_data(labels_layer)

        if self.method_choice_list.currentText() == "Correlation Filter":
            resulting_df = correlation_filter(reg_props, self.correlation_threshold.value)


            # replace previous table with new table containing only uncorrelating features
            set_features(labels_layer, resulting_df)
            # self.inactivate_correlation_boxes()

        print("Feature selection finished")
        show_table(self.viewer, labels_layer)


# TODO description of parameters
def get_uncorrelating_subselection(df_regprops, correlating_keys, kept_keys):
    """
    Returns a dataframe with only uncorrelating features based on given correlating keys
    and keys of features to be kept from the correlating groups
    """
    # getting all the feature keys that were correlating
    all_selectionkeys = []
    for keygroup in correlating_keys:
        all_selectionkeys += keygroup

    # finding out which keys to drop and dropping them
    dropkeys = [key for key in all_selectionkeys if key not in kept_keys]
    resulting_df = df_regprops.drop(dropkeys, axis=1)

    return resulting_df


def get_correlating_keys(df_regprops, threshold):
    """
    Returns sets of correlating features as lists of keys based on a dataframe
    containing region properties
    """
    import numpy as np

    # Actually finding the correlating features with pandas
    correlation_df = df_regprops.corr().abs()
    correlation_matrix = correlation_df.to_numpy()

    # using numpy to get the correlating features out of the matrix
    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
    masked_array = correlation_matrix * mask
    highly_corr = np.where(masked_array >= threshold)

    # Using sets as a datatype for easier agglomeration of the features
    # afterwards conversion back to list
    correlating_feats = [{i, j} for i, j in zip(highly_corr[0], highly_corr[1])]
    correlating_feats_agglo = agglomerate_corr_feats(correlating_feats)
    corr_ind_list = [sorted(list(i)) for i in correlating_feats_agglo]

    # getting the keys and then turning the indices into keys
    keys = df_regprops.keys()
    correlating_keys = [keys[ind].tolist() for ind in corr_ind_list]

    return correlating_keys


# TODO parameter description
def agglomerate_corr_feats(correlating_features_sets):
    """
    Returns sets of features which all correlate (if A and B correlate as well
    as B and C the group of ABC is returned as a set.) when given pairs of
    correlating features in the form of sets
    """

    new_sets = []
    for i in correlating_features_sets:
        unique_set = True

        for j in correlating_features_sets:
            intersect = i & j
            if len(intersect) > 0 and i != j:
                unique_set = False
                union = i | j
                if union not in new_sets:
                    new_sets.append(i | j)

        if unique_set:
            new_sets.append(i)

    if new_sets == correlating_features_sets:
        return new_sets
    else:
        return agglomerate_corr_feats(new_sets)

def correlation_filter(region_properties: pd.DataFrame, correlation_threshold):
    correlation_matrix = region_properties.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape)
            ,k=1
        ).astype(np.bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

    return region_properties.drop(to_drop, axis= 1)