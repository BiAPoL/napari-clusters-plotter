import os
import warnings
from enum import Enum, auto

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from napari_tools_menu import register_dock_widget
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QGuiApplication, QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._plotter_utilities import (
    clustered_plot_parameters,
    estimate_number_bins,
    make_cluster_overlay_img,
    unclustered_plot_parameters,
)
from ._Qt_code import (
    ICON_ROOT,
    MplCanvas,
    MyNavigationToolbar,
    button,
    collapsible_box,
    labels_container_and_selection,
    title,
)
from ._utilities import (
    add_column_to_layer_tabular_data,
    dask_cluster_image_timelapse,
    generate_cluster_image,
    get_layer_tabular_data,
    get_nice_colormap,
)

POINTER = "frame"

POSSIBLE_CLUSTER_IDS = ["KMEANS", "HDBSCAN", "MS", "GMM", "AC"]  # not including manual


class PlottingType(Enum):
    HISTOGRAM_2D = auto()
    SCATTER = auto()


@register_dock_widget(menu="Measurement > Plot measurements (ncp)")
@register_dock_widget(menu="Visualization > Plot measurements (ncp)")
class PlotterWidget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()

        self.cluster_ids = None
        self.viewer = napari_viewer

        # create a scroll area
        self.scrollArea = QScrollArea()
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumWidth(400)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.contents = QWidget()
        self.scrollArea.setWidget(self.contents)

        self.layout = QVBoxLayout(self.contents)
        self.layout.setAlignment(Qt.AlignTop)

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_labels_layer = None

        def manual_clustering_method(inside):
            inside = np.array(inside)  # leads to errors sometimes otherwise

            if self.analysed_layer is None or len(inside) == 0:
                return  # if nothing was plotted yet, leave
            clustering_ID = "MANUAL_CLUSTER_ID"

            features = get_layer_tabular_data(self.analysed_layer)

            modifiers = QGuiApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier and clustering_ID in features.keys():
                former_clusters = features[clustering_ID].to_numpy()
                former_clusters[inside] = np.max(former_clusters) + 1
                features.update(pd.DataFrame(former_clusters, columns=[clustering_ID]))
            else:
                features[clustering_ID] = inside.astype(int)
            add_column_to_layer_tabular_data(
                self.analysed_layer, clustering_ID, features[clustering_ID]
            )

            # update the dropdown, so that the "MANUAL_CLUSTER_ID" is added
            self.update_axes_and_clustering_id_lists()
            # set the selected item of the "clustering" combobox
            self.plot_cluster_id.setCurrentText(clustering_ID)

            # redraw the whole plot
            self.run(
                features,
                self.plot_x_axis_name,
                self.plot_y_axis_name,
                plot_cluster_name=clustering_ID,
            )
            self.labels_select.value.opacity = 0

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance
        self.graphics_widget = MplCanvas(
            self.figure, manual_clustering_method=manual_clustering_method
        )

        # Navigation widget
        self.toolbar = MyNavigationToolbar(self.graphics_widget, self)

        # Modify toolbar icons and some tooltips
        for action in self.toolbar.actions():
            text = action.text()
            if text == "Pan":
                action.setToolTip(
                    "Pan/Zoom: Left button pans; Right button zooms; Click once to activate; Click again to deactivate"
                )
            if text == "Zoom":
                action.setToolTip(
                    "Zoom to rectangle; Click once to activate; Click again to deactivate"
                )
            if len(text) > 0:  # i.e. not a separator item
                icon_path = os.path.join(ICON_ROOT, text + ".png")
                action.setIcon(QIcon(icon_path))

        # create a placeholder widget to hold the toolbar and graphics widget.
        graph_container = QWidget()
        graph_container.setMinimumHeight(300)
        graph_container.setLayout(QtWidgets.QVBoxLayout())
        graph_container.layout().addWidget(self.toolbar)
        graph_container.layout().addWidget(self.graphics_widget)

        self.layout.addWidget(graph_container, alignment=Qt.AlignTop)

        label_container = title("<b>Plotting</b>")

        # widget for the selection of labels layer
        (
            labels_layer_selection_container,
            self.labels_select,
        ) = labels_container_and_selection()

        # widget for the selection of axes
        axes_container = QWidget()
        axes_container.setLayout(QHBoxLayout())
        axes_container.layout().addWidget(QLabel("Axes"))
        self.plot_x_axis = QComboBox()
        self.plot_y_axis = QComboBox()
        axes_container.layout().addWidget(self.plot_x_axis)
        axes_container.layout().addWidget(self.plot_y_axis)

        # select from existing clustering-results
        cluster_container = QWidget()
        cluster_container.setLayout(QHBoxLayout())
        cluster_container.layout().addWidget(QLabel("Clustering"))
        self.plot_cluster_id = QComboBox()
        cluster_container.layout().addWidget(self.plot_cluster_id)

        # making buttons
        run_container, run_button = button("Plot")
        update_container, update_button = button("Update Axes/Clustering Options")

        ############################
        # Advanced plotting options
        ############################

        self.advanced_options_container = collapsible_box("Expand for advanced options")

        def replot():
            clustering_ID = None
            if self.cluster_ids is not None:
                clustering_ID = self.plot_cluster_id.currentText()

            features = get_layer_tabular_data(self.analysed_layer)

            # redraw the whole plot

            try:
                self.run(
                    features,
                    self.plot_x_axis_name,
                    self.plot_y_axis_name,
                    plot_cluster_name=clustering_ID,
                )

            except AttributeError:
                # In this case, replotting is not yet possible
                pass

        def checkbox_status_changed():
            replot()

        def plotting_type_changed():
            replot()

            if self.plotting_type.currentText() == PlottingType.HISTOGRAM_2D.name:
                self.bin_number_container.setVisible(True)
                self.log_scale_container.setVisible(True)
                with QSignalBlocker(self.plot_hide_non_selected):
                    self.plot_hide_non_selected.setChecked(True)
            else:
                self.bin_number_container.setVisible(False)
                self.log_scale_container.setVisible(False)

        def bin_number_set():
            replot()

        def bin_auto():
            self.bin_number_manual_container.setVisible(not self.bin_auto.isChecked())
            if self.bin_auto.isChecked():
                replot()

        # Combobox with plotting types
        combobox_plotting_container = QWidget()
        combobox_plotting_container.setLayout(QHBoxLayout())
        combobox_plotting_container.layout().addWidget(QLabel("Plotting type"))
        self.plotting_type = QComboBox()
        self.plotting_type.addItems(
            [PlottingType.SCATTER.name, PlottingType.HISTOGRAM_2D.name]
        )
        self.plotting_type.currentIndexChanged.connect(plotting_type_changed)
        combobox_plotting_container.layout().addWidget(self.plotting_type)

        self.bin_number_container = QWidget()
        self.bin_number_container.setLayout(QHBoxLayout())
        self.bin_number_container.layout().addWidget(QLabel("Number of bins"))

        self.bin_number_manual_container = QWidget()
        self.bin_number_manual_container.setLayout(QHBoxLayout())
        self.bin_number_spinner = QSpinBox()
        self.bin_number_spinner.setMinimum(1)
        self.bin_number_spinner.setMaximum(1000)
        self.bin_number_spinner.setValue(400)

        self.bin_number_manual_container.layout().addWidget(self.bin_number_spinner)
        self.bin_number_set = QPushButton("Set")
        self.bin_number_set.clicked.connect(bin_number_set)
        self.bin_number_manual_container.layout().addWidget(self.bin_number_set)

        self.bin_number_container.layout().addWidget(self.bin_number_manual_container)

        self.bin_auto = QCheckBox("Auto")
        self.bin_auto.setChecked(True)
        self.bin_auto.stateChanged.connect(bin_auto)
        self.bin_number_container.layout().addWidget(self.bin_auto)

        self.bin_number_manual_container.setVisible(False)
        self.bin_number_container.setVisible(False)

        self.log_scale_container = QWidget()
        self.log_scale_container.setLayout(QHBoxLayout())
        self.log_scale_container.layout().addWidget(QLabel("Log scale"))
        self.log_scale = QCheckBox("")
        self.log_scale.setChecked(False)
        self.log_scale.stateChanged.connect(replot)
        self.log_scale_container.layout().addWidget(self.log_scale)

        # Checkbox to hide non-selected clusters
        checkbox_container = QWidget()
        checkbox_container.setLayout(QHBoxLayout())
        checkbox_container.layout().addWidget(QLabel("Hide non-selected clusters"))
        self.plot_hide_non_selected = QCheckBox()
        self.plot_hide_non_selected.stateChanged.connect(checkbox_status_changed)
        checkbox_container.layout().addWidget(self.plot_hide_non_selected)

        self.advanced_options_container.addWidget(combobox_plotting_container)
        self.advanced_options_container.addWidget(self.log_scale_container)
        self.advanced_options_container.addWidget(self.bin_number_container)

        self.advanced_options_container.addWidget(checkbox_container)

        # adding all widgets to the layout
        self.layout.addWidget(label_container, alignment=Qt.AlignTop)
        self.layout.addWidget(labels_layer_selection_container, alignment=Qt.AlignTop)
        self.layout.addWidget(axes_container, alignment=Qt.AlignTop)
        self.layout.addWidget(cluster_container, alignment=Qt.AlignTop)

        self.layout.addWidget(self.advanced_options_container, alignment=Qt.AlignTop)

        self.layout.addWidget(update_container, alignment=Qt.AlignTop)
        self.layout.addWidget(run_container, alignment=Qt.AlignTop)
        self.layout.setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # adding spacing between fields for selecting two axes
        axes_container.layout().setSpacing(6)

        def run_clicked():
            if self.labels_select.value is None:
                warnings.warn("Please select labels layer!")
                return
            if get_layer_tabular_data(self.labels_select.value) is None:
                warnings.warn(
                    "No labels image with features/properties was selected! Consider doing measurements first."
                )
                return
            if (
                self.plot_x_axis.currentText() == ""
                or self.plot_y_axis.currentText() == ""
            ):
                warnings.warn(
                    "No axis(-es) was/were selected! If you cannot see anything in axes selection boxes, "
                    "but you have performed measurements/dimensionality reduction before, try clicking "
                    "Update Axes Selection Boxes"
                )
                return

            self.run(
                get_layer_tabular_data(self.labels_select.value),
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText(),
                self.plot_cluster_id.currentText(),
            )

        # takes care of case where this isn't set yet directly after init
        self.plot_cluster_name = None
        self.old_frame = None
        self.frame = self.viewer.dims.current_step[0]

        def frame_changed(event):
            if self.viewer.dims.ndim <= 3:
                return
            frame = event.value[0]
            if (not self.old_frame) or (self.old_frame != frame):
                if self.labels_select.value is None:
                    warnings.warn("Please select labels layer!")
                    return
                if get_layer_tabular_data(self.labels_select.value) is None:
                    warnings.warn(
                        "No labels image with features/properties was selected! Consider doing measurements first."
                    )
                    return
                if (
                    self.plot_x_axis.currentText() == ""
                    or self.plot_y_axis.currentText() == ""
                ):
                    warnings.warn(
                        "No axis(-es) was/were selected! If you cannot see anything in axes selection boxes, "
                        "but you have performed measurements/dimensionality reduction before, try clicking "
                        "Update Axes Selection Boxes"
                    )
                    return

                self.frame = frame

                self.run(
                    get_layer_tabular_data(self.labels_select.value),
                    self.plot_x_axis.currentText(),
                    self.plot_y_axis.currentText(),
                    self.plot_cluster_name,
                    redraw_cluster_image=False,
                )
            self.old_frame = frame

        # update axes combo boxes once a new label layer is selected
        self.labels_select.changed.connect(self.update_axes_and_clustering_id_lists)
        # depending on the select clustering ID, enable/disable the checkbox for hiding clusters
        self.plot_cluster_id.currentIndexChanged.connect(
            self.change_state_of_nonselected_checkbox
        )
        # replot if the clustering ID has changed
        self.plot_cluster_id.currentIndexChanged.connect(replot)

        # update axes combo boxes automatically if features of
        # layer are changed
        self.last_connected = None
        self.labels_select.changed.connect(self.activate_property_autoupdate)

        # update axes combo boxes once update button is clicked
        update_button.clicked.connect(self.update_axes_and_clustering_id_lists)

        # select what happens when the run button is clicked
        run_button.clicked.connect(run_clicked)

        self.viewer.dims.events.current_step.connect(frame_changed)

        self.update_axes_and_clustering_id_lists()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    def change_state_of_nonselected_checkbox(self):
        # enable the checkbox only if clustering is done manually or custom way/with a custom name,
        # i.e., algorithm name is not in POSSIBLE_CLUSTER_IDS
        enabled = (
            False
            if any(
                name in self.plot_cluster_id.currentText()
                for name in POSSIBLE_CLUSTER_IDS
            )
            else True
        )
        self.plot_hide_non_selected.setEnabled(enabled)
        # if HDBSCAN results are plotted, the checkbox will be disabled and checked,
        # because HDBSCAN is the only implemented algorithm that predicts noise points
        checked = "HDBSCAN" in self.plot_cluster_id.currentText()
        self.plot_hide_non_selected.setChecked(checked)

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                self.update_axes_and_clustering_id_lists
            )
        self.labels_select.value.events.properties.connect(
            self.update_axes_and_clustering_id_lists
        )
        self.last_connected = self.labels_select.value

    def update_axes_and_clustering_id_lists(self):
        selected_layer = self.labels_select.value

        former_x_axis = self.plot_x_axis.currentIndex()
        former_y_axis = self.plot_y_axis.currentIndex()
        former_cluster_id = self.plot_cluster_id.currentIndex()

        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.plot_x_axis.clear()
                self.plot_x_axis.addItems(list(features.keys()))
                self.plot_y_axis.clear()
                self.plot_y_axis.addItems(list(features.keys()))
                self.plot_cluster_id.clear()
                self.plot_cluster_id.addItem("")
                self.plot_cluster_id.addItems(
                    [
                        feature
                        for feature in list(features.keys())
                        if "CLUSTER" in feature
                    ]
                )
        self.plot_x_axis.setCurrentIndex(former_x_axis)
        self.plot_y_axis.setCurrentIndex(former_y_axis)
        self.plot_cluster_id.setCurrentIndex(former_cluster_id)

    def run(
        self,
        features: pd.DataFrame,
        plot_x_axis_name: str,
        plot_y_axis_name: str,
        plot_cluster_name=None,
        redraw_cluster_image=True,
        force_redraw: bool = False,
    ):
        """
        This function that runs after the run button is clicked.
        """

        if not self.isVisible() and force_redraw is False:
            # don't redraw in case the plot is invisible anyway
            return

        # check whether given axes names exist and if not don't redraw
        if (
            plot_x_axis_name not in features.columns
            or plot_y_axis_name not in features.columns
        ):
            print(
                "Selected measurements do not exist in layer's properties/features. The plot is not (re)drawn."
            )
            return

        self.data_x = features[plot_x_axis_name]
        self.data_y = features[plot_y_axis_name]
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.plot_cluster_name = plot_cluster_name
        self.analysed_layer = self.labels_select.value

        self.graphics_widget.reset()
        number_of_points = len(features)

        # if selected image is 4 dimensional, but does not contain frame column in its features
        # it will be considered to be tracking data, where all labels of the same track have
        # the same label, and each column represent track's features
        tracking_data = (
            len(self.analysed_layer.data.shape) == 4 and "frame" not in features.keys()
        )
        colors = get_nice_colormap()
        if (
            plot_cluster_name is not None
            and plot_cluster_name != "label"
            and plot_cluster_name in list(features.keys())
        ):
            if self.plot_hide_non_selected.isChecked():
                if "HDBSCAN" in plot_cluster_name:
                    pass  # this algorithm already predicts noise points as -1
                else:
                    features.loc[
                        features[plot_cluster_name] == 0, plot_cluster_name
                    ] = -1  # make unselected points to noise points

            # fill all prediction nan values with -1 -> turns them
            # into noise points
            self.label_ids = features["label"]
            self.cluster_ids = features[plot_cluster_name].fillna(-1)

            if len(self.analysed_layer.data.shape) == 4 and not tracking_data:
                frame_id = features[POINTER].tolist()
                current_frame = self.frame
            elif len(self.analysed_layer.data.shape) <= 3 or tracking_data:
                frame_id = None
                current_frame = None
            else:
                warnings.warn("Image dimensions too high for processing!")

            if self.plotting_type.currentText() == PlottingType.SCATTER.name:
                a, sizes, colors_plot = clustered_plot_parameters(
                    cluster_id=self.cluster_ids,
                    frame_id=frame_id,
                    current_frame=current_frame,
                    n_datapoints=number_of_points,
                    color_hex_list=colors,
                )

                self.graphics_widget.make_scatter_plot(
                    self.data_x, self.data_y, colors_plot, sizes, a
                )
            else:
                if self.bin_auto.isChecked():
                    number_bins = int(
                        np.max(
                            [
                                estimate_number_bins(self.data_x),
                                estimate_number_bins(self.data_y),
                            ]
                        )
                    )
                    self.bin_number_spinner.setValue(number_bins)
                else:
                    number_bins = int(self.bin_number_spinner.value())

                self.graphics_widget.make_2d_histogram(
                    self.data_x,
                    self.data_y,
                    colors,
                    bin_number=number_bins,
                    log_scale=self.log_scale.isChecked(),
                )

                rgb_img = make_cluster_overlay_img(
                    cluster_id=plot_cluster_name,
                    features=features,
                    feature_x=self.plot_x_axis_name,
                    feature_y=self.plot_y_axis_name,
                    colors=colors,
                    histogram_data=self.graphics_widget.histogram,
                    hide_first_cluster=self.plot_hide_non_selected.isChecked(),
                )
                xedges = self.graphics_widget.histogram[1]
                yedges = self.graphics_widget.histogram[2]

                self.graphics_widget.axes.imshow(
                    rgb_img,
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    origin="lower",
                    alpha=1,
                    aspect="auto",
                )
                self.graphics_widget.figure.canvas.draw_idle()

            self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
            self.graphics_widget.axes.set_ylabel(plot_y_axis_name)
            self.graphics_widget.match_napari_layout()

            from vispy.color import Color

            cmap = [Color(hex_name).RGBA.astype("float") / 255 for hex_name in colors]

            # generate dictionary mapping each prediction to its respective color
            # list cycling with  % introduced for all labels except hdbscan noise points (id = -1)
            cmap_dict = {
                int(prediction + 1): (
                    cmap[int(prediction) % len(cmap)]
                    if prediction >= 0
                    else [0, 0, 0, 0]
                )
                for prediction in np.unique(self.cluster_ids)
            }
            # take care of background label
            cmap_dict[0] = [0, 0, 0, 0]

            keep_selection = list(self.viewer.layers.selection)

            # Generating the cluster image
            if redraw_cluster_image:
                # depending on the dimensionality of the data
                # generate the cluster image
                if len(self.analysed_layer.data.shape) == 4:
                    if not tracking_data:
                        max_timepoint = features[POINTER].max() + 1
                        label_id_list_per_timepoint = [
                            features.loc[features[POINTER] == i]["label"].tolist()
                            for i in range(int(max_timepoint))
                        ]
                        prediction_lists_per_timepoint = [
                            features.loc[features[POINTER] == i][
                                plot_cluster_name
                            ].tolist()
                            for i in range(int(max_timepoint))
                        ]
                    else:
                        label_id_list_per_timepoint = [
                            features[plot_cluster_name].tolist()
                            for i in range(self.analysed_layer.data.shape[0])
                        ]
                        prediction_lists_per_timepoint = [
                            features[plot_cluster_name].tolist()
                            for i in range(self.analysed_layer.data.shape[0])
                        ]

                    cluster_image = dask_cluster_image_timelapse(
                        self.analysed_layer.data,
                        label_id_list_per_timepoint,
                        prediction_lists_per_timepoint,
                    )

                elif len(self.analysed_layer.data.shape) <= 3:
                    cluster_image = generate_cluster_image(
                        self.analysed_layer.data, self.label_ids, self.cluster_ids
                    )
                else:
                    warnings.warn("Image dimensions too high for processing!")
                    return

                # if the cluster image layer doesn't yet exist make it
                # otherwise just update it
                if (
                    self.visualized_labels_layer is None
                    or self.visualized_labels_layer not in self.viewer.layers
                ):
                    # visualising cluster image
                    self.visualized_labels_layer = self.viewer.add_labels(
                        cluster_image,  # self.analysed_layer.data
                        color=cmap_dict,  # cluster_id_dict
                        name="cluster_ids_in_space",
                        scale=self.labels_select.value.scale,
                    )
                else:
                    # updating data
                    self.visualized_labels_layer.data = cluster_image
                    self.visualized_labels_layer.color = cmap_dict

            self.viewer.layers.selection.clear()
            for s in keep_selection:
                self.viewer.layers.selection.add(s)

        else:
            if len(self.analysed_layer.data.shape) == 4 and not tracking_data:
                frame_id = features[POINTER].tolist()
                current_frame = self.frame
            elif len(self.analysed_layer.data.shape) <= 3 or tracking_data:
                frame_id = None
                current_frame = None
            else:
                warnings.warn("Image dimensions too high for processing!")

            if self.plotting_type.currentText() == PlottingType.SCATTER.name:
                a, sizes, colors_plot = unclustered_plot_parameters(
                    frame_id=frame_id,
                    current_frame=current_frame,
                    n_datapoints=number_of_points,
                )

                self.graphics_widget.make_scatter_plot(
                    self.data_x, self.data_y, colors_plot, sizes, a
                )
            else:
                if self.bin_auto.isChecked():
                    number_bins = int(
                        np.max(
                            [
                                estimate_number_bins(self.data_x),
                                estimate_number_bins(self.data_y),
                            ]
                        )
                    )
                    self.bin_number_spinner.setValue(number_bins)
                else:
                    number_bins = int(self.bin_number_spinner.value())

                self.graphics_widget.make_2d_histogram(
                    self.data_x,
                    self.data_y,
                    colors,
                    bin_number=number_bins,
                    log_scale=self.log_scale.isChecked(),
                )
            self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
            self.graphics_widget.axes.set_ylabel(plot_y_axis_name)
            self.graphics_widget.match_napari_layout()

            self.graphics_widget.draw()  # Only redraws when cluster is not manually selected
            # because manual selection already does that when selector is disconnected
