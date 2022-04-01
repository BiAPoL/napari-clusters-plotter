import warnings
from functools import partial

import pandas as pd
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

from ._utilities import restore_defaults, widgets_inactive

DEFAULTS = dict(correlation_threshold=0.95)


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
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))
        self.update_properties_list()

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

        # Analyse Correlation Button
        self.analyse_correlation_container = QWidget()
        self.analyse_correlation_container.setLayout(QHBoxLayout())
        self.analyse_correlation_button = QPushButton("Analyse Correlation")
        self.analyse_correlation_container.layout().addWidget(
            self.analyse_correlation_button
        )
        self.analyse_correlation_container.setVisible(False)

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

        def analyse_clicked():
            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            # measure correlation properties
            self.analyse_correlation()

            # adding widgets
            if self.correlating_keys is not None and len(self.correlating_keys) != 0:
                self.correlation_key_lists = [
                    QListWidget() for correlations in self.correlating_keys
                ]
                for widget, key_list in zip(
                    self.correlation_key_lists, self.correlating_keys
                ):
                    widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
                    widget.setGeometry(QRect(10, 10, 101, 291))
                    for i, p in enumerate(key_list):
                        item = QListWidgetItem(p)
                        widget.addItem(item)

                        if i == 0:
                            item.setSelected(True)

                self.correlation_containers = [
                    QWidget() for correlations in self.correlating_keys
                ]
                for i, widget in enumerate(self.correlation_containers):
                    widget.setLayout(QVBoxLayout())
                    widget.layout().addWidget(QLabel(f"Correlating Group #{i + 1}"))
                    widget.layout().addWidget(self.correlation_key_lists[i])
                    widget.setVisible(False)

                for container in self.correlation_containers:
                    self.layout().addWidget(container)
                self.layout().setSpacing(0)

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)
        defaults_button.clicked.connect(partial(restore_defaults, self, DEFAULTS))
        self.analyse_correlation_button.clicked.connect(analyse_clicked)

        # making sure checking before assignment doesn't lead to problems
        self.correlating_keys = None

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(method_container)
        self.layout().addWidget(self.correlation_threshold_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(defaults_container)
        self.layout().addWidget(run_widget)
        self.layout().addWidget(self.analyse_correlation_container)

        if self.correlating_keys is not None and len(self.correlating_keys) != 0:
            for container in self.correlation_containers:
                self.layout().addWidget(container)
            self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.method_choice_list.currentIndexChanged.connect(
            self.change_analyse_button
        )
        self.method_choice_list.currentIndexChanged.connect(
            self.change_correlation_threshold
        )
        self.method_choice_list.currentIndexChanged.connect(
            self.change_correlation_boxes
        )
        self.analyse_correlation_button.clicked.connect(
            self.change_correlation_boxes
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

    def change_analyse_button(self):
        widgets_inactive(
            self.analyse_correlation_container,
            active=self.method_choice_list.currentText() == "Correlation Filter",
        )

    def change_correlation_boxes(self):
        if self.correlating_keys is not None and len(self.correlating_keys) != 0:
            for widget in self.correlation_containers:
                widgets_inactive(
                    widget,
                    active=self.method_choice_list.currentText()
                    == "Correlation Filter",
                )

    def analyse_correlation(self):
        import numpy as np
        import pandas as pd

        # get thresholds and region properties from selected labels layer
        threshold = self.correlation_threshold.value
        self.update_properties_list()
        labels_layer = self.labels_select.value

        # convert properties to dataframe for further processing
        df_regprops = pd.DataFrame(labels_layer.properties)

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

        # remove label key and save correlating keys into self variable for later recall
        self.correlating_keys = [
            [key for key in keygroup if key != "label"] for keygroup in correlating_keys
        ]

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
    def run(self, labels_layer):
        print("Selected labels layer: " + str(labels_layer))

        # Turn properties from layer into a dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        if self.method_choice_list.currentText() == "Correlation Filter":
            resulting_df = self.get_uncorrelating_subselection(reg_props)

            # replace previous table with new table containing only uncorrelating features
            labels_layer.properties = resulting_df

        from ._utilities import show_table

        show_table(self.viewer, labels_layer)

        print("Feature selection finished")

    # TODO description of what this does
    def get_uncorrelating_subselection(self, df_regprops):
        kept_feats = []
        for widget in self.correlation_key_lists:
            kept_feats += [i.text() for i in widget.selectedItems()]

        # getting all the feature keys that were correlating
        all_selectionkeys = []
        for keygroup in self.correlating_keys:
            all_selectionkeys += keygroup

        # finding out which keys to drop and dropping them
        dropkeys = [key for key in all_selectionkeys if key not in kept_feats]
        resulting_df = df_regprops.drop(dropkeys, axis=1)

        return resulting_df


# TODO description of what this does
def agglomerate_corr_feats(correlating_features_sets):
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
