from enum import Enum, auto
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from biaplotter.plotter import ArtistType, CanvasWidget
from matplotlib.pyplot import cm as plt_colormaps
from nap_plot_tools.cmap import cat10_mod_cmap
from napari.utils.colormaps import ALL_COLORMAPS
from qtpy import uic
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ._algorithm_widget import BaseWidget


class PlottingType(Enum):
    HISTOGRAM = auto()
    SCATTER = auto()


class PlotterWidget(BaseWidget):
    """
    Widget for plotting data from selected layers in napari.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer to connect to.
    """

    plot_needs_update = Signal()

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self._setup_ui(napari_viewer)
        self._on_update_layer_selection(None)
        self._setup_callbacks()

        self.plot_needs_update.connect(self._replot)

    def _setup_ui(self, napari_viewer):
        """
        Helper function to set up the UI of the widget.
        """
        self.control_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "plotter_inputs.ui",
            self.control_widget,
        )

        self._selectors = {
            "x": self.control_widget.x_axis_box,
            "y": self.control_widget.y_axis_box,
            "hue": self.control_widget.hue_box,
        }
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.plotting_widget = CanvasWidget(napari_viewer, self)
        self.plotting_widget.active_artist = self.plotting_widget.artists[
            ArtistType.SCATTER
        ]

        # Add plot and options as widgets
        self.layout.addWidget(self.plotting_widget)
        self.layout.addWidget(self.control_widget)

        # Setting of Widget options
        self.hue: QComboBox = self.control_widget.hue_box

        self.control_widget.plot_type_box.addItems(
            [PlottingType.SCATTER.name, PlottingType.HISTOGRAM.name]
        )

        self.control_widget.cmap_box.addItems(list(ALL_COLORMAPS.keys()))
        self.control_widget.cmap_box.setCurrentIndex(
            np.argwhere(np.array(list(ALL_COLORMAPS.keys())) == "magma")[0][0]
        )

        # Setting Visibility Defaults
        self.control_widget.manual_bins_container.setVisible(False)
        self.control_widget.bins_settings_container.setVisible(False)
        self.control_widget.log_scale_container.setVisible(False)

    def _setup_callbacks(self):
        """
        Set up the callbacks for the widget.
        """

        # Connect all necessary functions to the replot
        connections_to_replot = [
            (
                self.control_widget.plot_type_box.currentIndexChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.set_bins_button.clicked,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.auto_bins_checkbox.stateChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.log_scale_checkbutton.stateChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.non_selected_checkbutton.stateChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.cmap_box.currentIndexChanged,
                self.plot_needs_update.emit,
            ),
        ]

        for signal, callback in connections_to_replot:
            signal.connect(callback)

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].currentTextChanged.connect(
                self.plot_needs_update.emit
            )

        self.viewer.layers.selection.events.changed.connect(
            self._on_update_layer_selection
        )

        # connect frame change to alpha update
        self.viewer.dims.events.current_step.connect(self._on_frame_changed)

        # reset the coloring of the selected layer
        self.control_widget.reset_button.clicked.connect(self._reset)

        # connect data selection in plot to layer coloring update
        for selector in self.plotting_widget.selectors.values():
            selector.selection_applied_signal.connect(self._on_finish_draw)

    def _on_finish_draw(self, color_indices: np.ndarray):
        """
        Called when user finsihes drawing. Will change the hue combo box to the
        feature 'MANUAL_CLUSTER_ID', which then triggers a redraw.
        """

        # if the hue axis is not set to MANUAL_CLUSTER_ID, set it to that
        # otherwise replot the data

        features = self._get_features()
        for layer in self.viewer.layers.selection:
            layer_indices = features[features["layer"] == layer.name].index

            # store latest cluster indeces in the features table
            layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                color_indices[layer_indices]
            ).astype("category")

        if self.hue_axis != "MANUAL_CLUSTER_ID":
            self.hue_axis = "MANUAL_CLUSTER_ID"

        self.plot_needs_update.emit()

    def _replot(self):
        """
        Replot the data with the current settings.
        """

        # if no x or y axis is selected, return
        if self.x_axis == "" or self.y_axis == "":
            return

        # retrieve the data from the selected layers
        features = self._get_features()
        x_data = features[self.x_axis].values
        y_data = features[self.y_axis].values

        # check hue axis for categorical data
        if self.hue_axis in self.categorical_columns:
            self.plotting_widget.active_artist.overlay_colormap = (
                cat10_mod_cmap
            )
        else:
            self.plotting_widget.active_artist.overlay_colormap = (
                plt_colormaps.magma
            )

        # set the data and color indices in the active artist
        active_artist = self.plotting_widget.active_artist
        active_artist.data = np.stack([x_data, y_data], axis=1)
        active_artist.color_indices = features[self.hue_axis].to_numpy()

        self._color_layer_by_value()

        # this makes sure that previously drawn clusters are preserved
        # when a layer is re-selected or different features are plotted
        # if "MANUAL_CLUSTER_ID" in features.columns:
        #     self.plotting_widget.active_artist.color_indices = features[
        #         "MANUAL_CLUSTER_ID"
        #     ].to_numpy()

    def _on_frame_changed(self, event: napari.utils.events.Event):
        """
        Called when the frame changes. Updates the alpha values of the points.
        """

        if "frame" in self._get_features().columns:
            current_step = self.viewer.dims.current_step[0]
            alpha = np.asarray(
                self._get_features()["frame"] == current_step, dtype=float
            )
            size = np.ones(len(alpha)) * 50

            index_out_of_frame = alpha == 0
            alpha[index_out_of_frame] = 0.25
            size[index_out_of_frame] = 35
            self.plotting_widget.active_artist.alpha = alpha
            self.plotting_widget.active_artist.size = size

    def _checkbox_status_changed(self):
        self._replot()

    def _plotting_type_changed(
        self,
    ):  # TODO NEED TO ADD WHICH VARIABLE STORES THE TYPE
        if (
            self.control_widget.plot_type_box.currentText()
            == PlottingType.HISTOGRAM.name
        ):
            self.control_widget.bins_settings_container.setVisible(True)
            self.control_widget.log_scale_container.setVisible(True)
        elif (
            self.control_widget.plot_type_box.currentText()
            == PlottingType.SCATTER.name
        ):
            self.control_widget.bins_settings_container.setVisible(False)
            self.control_widget.log_scale_container.setVisible(False)

        self._replot()

    def _bin_number_set(self):
        self._replot()

    def _bin_auto(self):
        self.control_widget.manual_bins_container.setVisible(
            not self.control_widget.auto_bins_checkbox.isChecked()
        )
        if self.control_widget.auto_bins_checkbox.isChecked():
            self._replot()

    # Connecting the widgets to actual object variables:
    # using getters and setters for flexibility
    @property
    def log_scale(self):
        return self.control_widget.log_scale_checkbutton.isChecked()

    @log_scale.setter
    def log_scale(self, val: bool):
        self.control_widget.log_scale_checkbutton.setChecked(val)

    @property
    def automatic_bins(self):
        return self.control_widget.auto_bins_checkbox.isChecked()

    @automatic_bins.setter
    def automatic_bins(self, val: bool):
        self.control_widget.auto_bins_checkbox.setChecked(val)

    @property
    def bin_number(self):
        return self.control_widget.n_bins_box.value()

    @property
    def hide_non_selected(self):
        return self.control_widget.non_selected_checkbutton.isChecked()

    @hide_non_selected.setter
    def hide_non_selected(self, val: bool):
        self.control_widget.non_selected_checkbutton.setChecked(val)

    @property
    def colormap_plot(self):
        return self.control_widget.cmap_box.currentText()

    @property
    def plotting_type(self):
        return self.control_widget.plot_type_box.currentText()

    @plotting_type.setter
    def plotting_type(self, plot_type):
        if plot_type in PlottingType.__members__:
            self.control_widget.plot_type_box.setCurrentText(plot_type)

    @property
    def x_axis(self):
        return self.control_widget.x_axis_box.currentText()

    @x_axis.setter
    def x_axis(self, column: str):
        self.control_widget.x_axis_box.setCurrentText(column)
        self._replot()

    @property
    def y_axis(self):
        return self.control_widget.y_axis_box.currentText()

    @y_axis.setter
    def y_axis(self, column: str):
        self.control_widget.y_axis_box.setCurrentText(column)
        self._replot()

    @property
    def hue_axis(self):
        return self.control_widget.hue_box.currentText()

    @hue_axis.setter
    def hue_axis(self, column: str):
        """
        Set the hue axis to the given value.
        """
        # check if the column is in the common columns
        if column not in self.common_columns:
            raise ValueError(
                f"{column} is not in the features: {self.common_columns}"
            )
        self.control_widget.hue_box.setCurrentText(column)

    def _on_update_layer_selection(
        self, event: napari.utils.events.Event
    ) -> None:
        """
        Called when the layer selection changes. Updates the layers attribute.
        """
        # don't do anything if no layer is selected
        if self.n_selected_layers == 0:
            return

        # check if the selected layers are of the correct type
        selected_layer_types = [
            type(layer) for layer in self.viewer.layers.selection
        ]
        for layer_type in selected_layer_types:
            if layer_type not in self.input_layer_types:
                return

        # check if all selected layers are of the same type
        if len(set(selected_layer_types)) > 1:
            return

        # insert 'MANUAL_CLUSTER_ID' column if it doesn't exist
        for layer in self.viewer.layers.selection:
            if "MANUAL_CLUSTER_ID" not in layer.features.columns:
                layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                    np.zeros(len(layer.features), dtype=np.int32)
                ).astype("category")

        self.layers = list(self.viewer.layers.selection)
        self._update_feature_selection(None)

        for layer in self.layers:
            layer.events.features.connect(self._update_feature_selection)

    def _update_feature_selection(
        self, event: napari.utils.events.Event
    ) -> None:
        """
        Update the features in the dropdowns.
        """
        self.blockSignals(True)
        current_x = self.x_axis
        current_y = self.y_axis
        current_hue = self.hue_axis

        # get the common columns between the selected layers
        # and the columns that are not categorical
        continuous_features = sorted(
            [
                col
                for col in self.common_columns
                if col not in self.categorical_columns
            ]
        )

        for dim, current_value in zip(
            ["x", "y", "hue"], [current_x, current_y, current_hue]
        ):
            # block selector changed signals until all items added
            selector = self._selectors[dim]
            selector.blockSignals(True)
            selector.clear()

            if dim in ["x", "y"]:
                selector.addItems(continuous_features)
            elif dim == "hue":
                selector.addItems(sorted(self.common_columns))
                self._set_categorical_column_styles(
                    selector, self.categorical_columns
                )

            # set the previous values if they are still available
            if current_value in self.common_columns:
                selector.setCurrentText(current_value)

            selector.blockSignals(False)

        self.blockSignals(False)
        self.plot_needs_update.emit()

    def _set_categorical_column_styles(self, selector, categorical_columns):
        """Highlight categorical columns and set tooltips."""
        for feature in categorical_columns:
            index = selector.findText(feature)
            if index != -1:  # Ensure the feature exists in the dropdown
                selector.setItemData(
                    index, QColor("darkOrange"), Qt.BackgroundRole
                )
                selector.setItemData(
                    index, "Categorical Column", Qt.ToolTipRole
                )

    def _color_layer_by_value(self):
        """
        Color the selected layer according to the color indices.
        """

        features = self._get_features()
        color_indices = self.plotting_widget.active_artist.color_indices
        norm = self.plotting_widget.active_artist._get_normalization(
            color_indices
        )
        colors = self.plotting_widget.active_artist._get_rgba_colors(
            color_indices, norm
        )

        for selected_layer in self.viewer.layers.selection:
            layer_indices = features[
                features["layer"] == selected_layer.name
            ].index
            _apply_layer_color(selected_layer, colors[layer_indices])

            # store latest cluster indeces in the features table
            if self.hue_axis == "MANUAL_CLUSTER_ID":
                selected_layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                    color_indices[layer_indices]
                ).astype("category")

    def _reset(self):
        """
        Reset the selection in the current plotting widget.
        """
        self.plotting_widget.active_artist.color_indices = np.zeros(
            len(self._get_features())
        )
        self._color_layer_by_value()


def _apply_layer_color(layer, colors):
    """
    Apply colors to the layer based on the layer type.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to color.

    colors : np.ndarray
        The color array (Nx4).
    """
    from napari.utils import DirectLabelColormap

    if isinstance(layer, napari.layers.Points):
        layer.face_color = colors

    elif isinstance(layer, napari.layers.Vectors):
        layer.edge_color = colors

    elif isinstance(layer, napari.layers.Surface):
        layer.vertex_colors = colors

    elif isinstance(layer, napari.layers.Shapes):
        layer.face_color = colors

    elif isinstance(layer, napari.layers.Labels):

        colors = np.insert(colors, 0, [0, 0, 0, 0], axis=0)
        color_dict = dict(zip(np.unique(layer.data), colors))

        # Insert default colors for labels that are not in the color_dict
        # Relevant for non-sequential label images
        if max(color_dict.keys()) > len(colors):
            for i in range(1, max(color_dict.keys()) - 1):
                color_dict[i] = [0, 0, 0, 0]
        # Add a color for the background at the first index
        layer.colormap = DirectLabelColormap(color_dict=color_dict)

    layer.refresh()
