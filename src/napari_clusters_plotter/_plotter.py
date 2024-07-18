import os
import warnings
from enum import Enum, auto

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from napari.layers import Image, Labels, Layer, Points, Surface
from napari.utils.colormaps import ALL_COLORMAPS
from napari_tools_menu import register_dock_widget
from qtpy import QtWidgets
from qtpy.QtCore import Qt
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
    apply_cluster_colors_to_bars,
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
    create_options_dropdown,
    layer_container_and_selection,
    title,
)
from ._utilities import (
    _POINTER,
    add_column_to_layer_tabular_data,
    generate_cluster_image,
    generate_cluster_surface,
    get_layer_tabular_data,
)

POSSIBLE_CLUSTER_IDS = ["KMEANS", "HDBSCAN", "MS", "GMM", "AC"]  # not including manual


class PlottingType(Enum):
    HISTOGRAM = auto()
    SCATTER = auto()


@register_dock_widget(menu="Measurement > Plot measurements (ncp)")
@register_dock_widget(menu="Visualization > Plot measurements (ncp)")
class PlotterWidget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()

        self.layer_coloring_functions = {
            Labels: generate_cluster_image,
            Surface: generate_cluster_surface,
        }

        self.cluster_ids = None
        self.visualized_layer = None
        self.viewer = napari_viewer

        # create a scroll area
        self.scrollArea = QScrollArea()
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumWidth(450)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.contents = QWidget()
        self.scrollArea.setWidget(self.contents)

        self.layout = QVBoxLayout(self.contents)
        self.layout.setAlignment(Qt.AlignTop)

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_layer = None

        def manual_clustering_method(inside):
            inside = np.array(inside)  # leads to errors sometimes otherwise

            if self.analysed_layer is None or len(inside) == 0:
                return  # if nothing was plotted yet, leave
            clustering_ID = "MANUAL_CLUSTER_ID"

            features = get_layer_tabular_data(self.analysed_layer)

            modifiers = QGuiApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier and clustering_ID in features.keys():
                features[clustering_ID] = (
                    features[clustering_ID]
                    .mask(
                        inside,
                        other=features[clustering_ID].max() + 1,
                    )
                    .to_numpy()
                )
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
            if isinstance(self.analysed_layer, Labels):
                self.layer_select.value.opacity = 0.2

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

        # widget for the selection of layer
        (
            layer_selection_container,
            self.layer_select,
        ) = layer_container_and_selection(viewer=self.viewer)

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

        self.advanced_options_container_box = collapsible_box(
            "Expand for advanced options"
        )
        self.advanced_options_container = QWidget(self)
        self.advanced_options_container.setLayout(QVBoxLayout())
        self.advanced_options_container_box.addWidget(self.advanced_options_container)
        self.advanced_options_container.layout().setSpacing(0)
        self.advanced_options_container.layout().setContentsMargins(0, 0, 0, 0)

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
            if self.plotting_type.currentText() == PlottingType.HISTOGRAM.name:
                self.bin_number_container.setVisible(True)
                self.log_scale_container.setVisible(True)
                self.plot_hide_non_selected.setChecked(True)
                self.colormap_container.setVisible(True)
            else:
                self.bin_number_container.setVisible(False)
                self.log_scale_container.setVisible(False)
                self.colormap_container.setVisible(False)
            replot()

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
            [PlottingType.SCATTER.name, PlottingType.HISTOGRAM.name]
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

        self.log_scale_container.setVisible(False)

        # Checkbox to hide non-selected clusters
        self.hide_nonselected_checkbox_container = QWidget()
        self.hide_nonselected_checkbox_container.setLayout(QHBoxLayout())
        self.hide_nonselected_checkbox_container.layout().addWidget(
            QLabel("Hide non-selected clusters")
        )
        self.plot_hide_non_selected = QCheckBox()
        self.plot_hide_non_selected.setToolTip("Enabled only for manual clustering")
        self.plot_hide_non_selected.stateChanged.connect(checkbox_status_changed)
        self.hide_nonselected_checkbox_container.layout().addWidget(
            self.plot_hide_non_selected
        )

        self.advanced_options_container.layout().addWidget(combobox_plotting_container)
        self.advanced_options_container.layout().addWidget(self.log_scale_container)
        self.advanced_options_container.layout().addWidget(self.bin_number_container)
        self.advanced_options_container.layout().addWidget(
            self.hide_nonselected_checkbox_container
        )

        # selection of possible colormaps for 2D histogram
        self.colormap_container, self.colormap_dropdown = create_options_dropdown(
            name="Colormap",
            value="magma",
            options={"choices": list(ALL_COLORMAPS.keys())},
            label="Colormap",
        )
        self.colormap_container.setVisible(False)
        self.colormap_dropdown.native.currentIndexChanged.connect(replot)
        self.advanced_options_container.layout().addWidget(self.colormap_container)

        # adding all widgets to the layout
        self.layout.addWidget(label_container, alignment=Qt.AlignTop)
        self.layout.addWidget(layer_selection_container, alignment=Qt.AlignTop)
        self.layout.addWidget(axes_container, alignment=Qt.AlignTop)
        self.layout.addWidget(cluster_container, alignment=Qt.AlignTop)

        self.layout.addWidget(
            self.advanced_options_container_box, alignment=Qt.AlignTop
        )

        self.layout.addWidget(update_container, alignment=Qt.AlignTop)
        self.layout.addWidget(run_container, alignment=Qt.AlignTop)
        self.layout.setSpacing(0)

        # go through all widgets and change spacing
        for widget_list in [self.layout, self.advanced_options_container.layout()]:
            for i in range(widget_list.count()):
                item = widget_list.itemAt(i).widget()
                if item.layout() is not None:
                    item.layout().setSpacing(0)
                    item.layout().setContentsMargins(3, 3, 3, 3)

        # adding spacing between fields for selecting two axes
        axes_container.layout().setSpacing(6)

        def run_clicked():
            if self.layer_select.value is None:
                warnings.warn("Please select labels layer!")
                return
            if get_layer_tabular_data(self.layer_select.value) is None:
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
                get_layer_tabular_data(self.layer_select.value),
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText(),
                self.plot_cluster_id.currentText(),
            )

        # takes care of case where this isn't set yet directly after init
        self.plot_cluster_name = None
        self.old_frame = None
        # Assume time is the first axis
        self.frame = self.viewer.dims.current_step[0]

        def frame_changed(event):
            if self.viewer.dims.ndim <= 3:
                return
            frame = event.value[0]
            if (not self.old_frame) or (self.old_frame != frame):
                if self.layer_select.value is None:
                    warnings.warn("Please select labels layer!")
                    return
                if get_layer_tabular_data(self.layer_select.value) is None:
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
                    get_layer_tabular_data(self.layer_select.value),
                    self.plot_x_axis.currentText(),
                    self.plot_y_axis.currentText(),
                    self.plot_cluster_name,
                    redraw_cluster_image=False,
                )
            self.old_frame = frame

        # update axes combo boxes once a new label layer is selected
        self.layer_select.changed.connect(self.update_axes_and_clustering_id_lists)
        # depending on the select clustering ID, enable/disable the checkbox for hiding clusters
        self.plot_cluster_id.currentIndexChanged.connect(
            self.change_state_of_nonselected_checkbox
        )

        # update axes combo boxes automatically if features of
        # layer are changed
        self.last_connected = None
        self.layer_select.changed.connect(self.activate_property_autoupdate)

        # update axes combo boxes once update button is clicked
        update_button.clicked.connect(self.update_axes_and_clustering_id_lists)

        # select what happens when the run button is clicked
        run_button.clicked.connect(run_clicked)

        self.viewer.dims.events.current_step.connect(self.frame_changed)

        self.update_axes_and_clustering_id_lists()

    def frame_changed(self, event):
        if self.viewer.dims.ndim <= 3:
            return
        frame = event.value[0]
        if (not self.old_frame) or (self.old_frame != frame):
            if self.layer_select.value is None:
                warnings.warn("Please select labels layer!")
                return
            if get_layer_tabular_data(self.layer_select.value) is None:
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
                get_layer_tabular_data(self.layer_select.value),
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText(),
                self.plot_cluster_name,
                redraw_cluster_image=False,
            )
        self.old_frame = frame

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.layer_select.reset_choices(event)

    def change_state_of_nonselected_checkbox(self):
        # make the checkbox visible only if clustering is done manually
        visible = (
            True
            if "MANUAL_CLUSTER_ID" in self.plot_cluster_id.currentText()
            or self.plot_cluster_id.currentText() == ""
            else False
        )
        self.hide_nonselected_checkbox_container.setVisible(visible)

        if any(
            name in self.plot_cluster_id.currentText() for name in POSSIBLE_CLUSTER_IDS
        ):
            self.plot_hide_non_selected.setChecked(False)

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(
                self.update_axes_and_clustering_id_lists
            )
        if not isinstance(self.layer_select.value, Image):
            self.layer_select.value.events.properties.connect(
                self.update_axes_and_clustering_id_lists
            )
        self.last_connected = self.layer_select.value

    def update_axes_and_clustering_id_lists(self):
        selected_layer = self.layer_select.value

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
        from napari.layers import Labels, Surface
        from vispy.color import Color

        from ._utilities import _is_pseudo_tracking, get_nice_colormap

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
        self.analysed_layer = self.layer_select.value

        self.graphics_widget.reset()

        self.graphics_widget.selected_colormap = self.colormap_dropdown.value

        number_of_points = len(features)

        # if selected image is 4 dimensional, but does not contain frame column in its features
        # it will be considered to be tracking data, where all labels of the same track have
        # the same label, and each column represent track's features
        tracking_data = _is_pseudo_tracking(self.analysed_layer)
        colors = get_nice_colormap()

        frame_id = None
        current_frame = None
        if isinstance(self.analysed_layer, Labels):
            if len(self.analysed_layer.data.shape) == 4 and not tracking_data:
                frame_id = features[_POINTER].tolist()
                current_frame = self.frame
            elif len(self.analysed_layer.data.shape) <= 3 or tracking_data:
                pass
            else:
                warnings.warn("Image dimensions too high for processing!")
        elif isinstance(self.analysed_layer, Surface):
            pass
        elif isinstance(self.analysed_layer, Points):
            pass
        else:
            warnings.warn(f"Layer {type(self.analysed_layer)} not supported")

        # check if 'frame' is in columns and enable frame highlighting if it is
        if "frame" in self.analysed_layer.features.columns:
            frame_id = features[_POINTER].tolist()
            current_frame = self.frame

        if (
            plot_cluster_name is not None
            and plot_cluster_name != "label"
            and plot_cluster_name in list(features.keys())
        ):
            if self.plot_hide_non_selected.isChecked():
                features.loc[features[plot_cluster_name] == 0, plot_cluster_name] = (
                    -1
                )  # make unselected points to noise points

            # fill all prediction nan values with -1 -> turns them
            # into noise points
            if "label" in features.keys():
                self.label_ids = features["label"]
            self.cluster_ids = features[plot_cluster_name].fillna(-1)

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

                self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
                self.graphics_widget.axes.set_ylabel(plot_y_axis_name)
            else:
                if self.bin_auto.isChecked():
                    if plot_x_axis_name == plot_y_axis_name:
                        number_bins = int(estimate_number_bins(self.data_x))
                    else:
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

                # if both axes are the same, plot 1D histogram
                if plot_x_axis_name == plot_y_axis_name:
                    self.graphics_widget.make_1d_histogram(
                        self.data_x,
                        bin_number=number_bins,
                        log_scale=self.log_scale.isChecked(),
                    )
                    # update bar colors to cluster ids
                    self.graphics_widget.axes = apply_cluster_colors_to_bars(
                        self.graphics_widget.axes,
                        cluster_name=plot_cluster_name,
                        features=features,
                        number_bins=number_bins,
                        feature_x=self.plot_x_axis_name,
                        colors=colors,
                    )
                    self.graphics_widget.figure.canvas.draw_idle()
                    self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
                    self.graphics_widget.axes.set_ylabel("frequency")
                else:
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
                self._update_cluster_image(
                    is_tracking_data=tracking_data,
                    plot_cluster_name=plot_cluster_name,
                    cmap_dict=cmap_dict,
                )

            self.viewer.layers.selection.clear()
            for s in keep_selection:
                self.viewer.layers.selection.add(s)

        else:
            if self.plotting_type.currentText() == PlottingType.SCATTER.name:
                a, sizes, colors_plot = unclustered_plot_parameters(
                    frame_id=frame_id,
                    current_frame=current_frame,
                    n_datapoints=number_of_points,
                )

                self.graphics_widget.make_scatter_plot(
                    self.data_x, self.data_y, colors_plot, sizes, a
                )

                self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
                self.graphics_widget.axes.set_ylabel(plot_y_axis_name)
            else:
                if self.bin_auto.isChecked():
                    if plot_x_axis_name == plot_y_axis_name:
                        number_bins = int(estimate_number_bins(self.data_x))
                    else:
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

                # if both axes are the same, plot 1D histogram
                if plot_x_axis_name == plot_y_axis_name:
                    self.graphics_widget.make_1d_histogram(
                        self.data_x,
                        bin_number=number_bins,
                        log_scale=self.log_scale.isChecked(),
                    )
                    self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
                    self.graphics_widget.axes.set_ylabel("frequency")
                else:
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
            self.graphics_widget.draw()

        if self.graphics_widget.last_xy_labels != (plot_x_axis_name, plot_y_axis_name):
            # Additional redraw in case axis have changed, otherwise y-axis may not get updated. General redraw would
            # resets the zoom, which needs to be avoided.
            self.graphics_widget.draw()

        self.graphics_widget.reset_zoom()

    def _update_cluster_image(
        self, is_tracking_data: bool, plot_cluster_name: str, cmap_dict: dict
    ):
        # if the cluster image layer doesn't yet exist make it
        self.visualized_layer = self._draw_cluster_image(
            is_tracking_data=is_tracking_data,
            plot_cluster_name=plot_cluster_name,
            cluster_ids=self.cluster_ids,
            cmap_dict=cmap_dict,
        )
        if (
            self.visualized_layer is None
            or self.visualized_layer.name not in self.viewer.layers
        ):
            self.viewer.add_layer(self.visualized_layer)
        else:
            layer_in_viewer = self.viewer.layers[self.visualized_layer.name]
            layer_in_viewer.data = self.visualized_layer.data
            if isinstance(self.visualized_layer, Points):
                layer_in_viewer.face_color = self.visualized_layer.face_color
            elif isinstance(self.visualized_layer, Surface):
                layer_in_viewer.colormap = self.visualized_layer.colormap
                layer_in_viewer.contrast_limits = self.visualized_layer.contrast_limits
            elif isinstance(self.visualized_layer, Labels):
                layer_in_viewer.color = self.visualized_layer.color
            else:
                print("Update failed")

    def _draw_cluster_image(
        self,
        is_tracking_data: bool,
        plot_cluster_name: str,
        cluster_ids,
        cmap_dict=None,
    ) -> Layer:
        from matplotlib.colors import to_rgba_array

        from ._utilities import (
            generate_cluster_4d_labels,
            generate_cluster_image,
            generate_cluster_surface,
            generate_cluster_tracks,
            get_nice_colormap,
            get_surface_color_map,
        )

        """
        Generate the cluster image layer.
        """
        nice_colormap = get_nice_colormap()
        napari_colormap = get_surface_color_map(max(cluster_ids))

        if (
            isinstance(self.analysed_layer, Labels)
            and len(self.analysed_layer.data.shape) == 4
            and not is_tracking_data
        ):
            cluster_data = generate_cluster_4d_labels(
                self.analysed_layer, plot_cluster_name
            )

            cluster_layer = Layer.create(
                cluster_data,
                {
                    "color": cmap_dict,
                    "name": "cluster_ids_in_space",
                    "scale": self.layer_select.value.scale,
                },
            )

        elif (
            isinstance(self.analysed_layer, Labels)
            and len(self.analysed_layer.data.shape) == 4
            and is_tracking_data
        ):
            cluster_data = generate_cluster_tracks(
                self.analysed_layer, plot_cluster_name
            )

            cluster_layer = Layer.create(
                cluster_data,
                {
                    "color": cmap_dict,
                    "name": "cluster_ids_in_space",
                    "scale": self.layer_select.value.scale,
                },
            )

        elif isinstance(self.analysed_layer, Surface):
            cluster_data = generate_cluster_surface(
                self.analysed_layer.data, self.cluster_ids
            )

            cluster_layer = Layer.create(
                cluster_data,
                {
                    "contrast_limits": [0, self.cluster_ids.max() + 1],
                    "colormap": napari_colormap,
                    "name": "cluster_ids_in_space",
                    "scale": self.layer_select.value.scale,
                },
                "surface",
            )

        elif isinstance(self.analysed_layer, Points):
            face_colors = to_rgba_array(np.asarray(nice_colormap)[cluster_ids])
            cluster_layer = Layer.create(
                self.analysed_layer.data,
                {
                    "face_color": face_colors,
                    "size": self.layer_select.value.size,
                    "name": "cluster_ids_in_space",
                    "scale": self.layer_select.value.scale,
                },
                "points",
            )
        elif len(self.analysed_layer.data.shape) <= 3:
            cluster_data = generate_cluster_image(
                self.analysed_layer.data, self.label_ids, self.cluster_ids
            ).astype(int)
            cluster_layer = Layer.create(
                cluster_data,
                {
                    "color": cmap_dict,
                    "name": "cluster_ids_in_space",
                    "scale": self.layer_select.value.scale,
                },
                "labels",
            )
        else:
            warnings.warn("Image dimensions too high for processing!")
            return

        return cluster_layer
