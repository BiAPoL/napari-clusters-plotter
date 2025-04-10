from enum import Enum, auto
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from biaplotter.plotter import ArtistType, CanvasWidget
from napari.utils.colormaps import ALL_COLORMAPS
from qtpy import uic
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget
from nap_plot_tools.cmap import cat10_mod_cmap
from matplotlib.pyplot import cm as plt_colormaps

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

        # reset the coloring of the selected layer
        self.control_widget.reset_button.clicked.connect(self._reset)

        # connect data selection in plot to layer coloring update
        for selector in self.plotting_widget.selectors.values():
            selector.selection_applied_signal.connect(
                self._on_finish_draw)

    def _on_finish_draw(self, color_indices: np.ndarray):
        """
        Called when user finsihes drawing. Will change the hue combo box to the
        feature 'MANUAL_CLUSTER_ID', which then triggers a redraw.
        """

        # if the hue axis is not set to MANUAL_CLUSTER_ID, set it to that
        # otherwise replot the data

        features = self._get_features()
        for layer in self.viewer.layers.selection:
            layer_indices = features[
                features["layer"] == layer.name
            ].index

            # store latest cluster indeces in the features table
            layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                color_indices[layer_indices]).astype("category")

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
            self.plotting_widget.active_artist.overlay_colormap = cat10_mod_cmap
        else:
            self.plotting_widget.active_artist.overlay_colormap = plt_colormaps.magma

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

        # block selector changed signals until all items added
        for dim in ["x", "y", "hue"]:
            self._selectors[dim].blockSignals(True)

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].clear()

        # get the common columns between the selected layers
        # and the columns that are not categorical
        xyfeatures_to_add = sorted(
            [
                col for col in self.common_columns
                if col not in self.categorical_columns
                ]
        )
        for dim in ["x", "y"]:
            self._selectors[dim].addItems(xyfeatures_to_add)

        # populate hue selector with all possible columns
        self._selectors['hue'].addItems(self.common_columns)

        # set the previous values if they are still available
        for dim, value in zip(
            ["x", "y", "hue"], [current_x, current_y, current_hue]
        ):
            if value in self.common_columns:
                self._selectors[dim].setCurrentText(value)

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].blockSignals(False)

        self.blockSignals(False)
        self.plot_needs_update.emit()

    def _color_layer_by_value(self):
        """
        Color the selected layer according to the color indices.
        """

        features = self._get_features()
        color_indices = self.plotting_widget.active_artist.color_indices
        norm = self.plotting_widget.active_artist._get_normalization(color_indices)
        colors = self.plotting_widget.active_artist._get_rgba_colors(color_indices, norm)

        for selected_layer in self.viewer.layers.selection:
            layer_indices = features[
                features["layer"] == selected_layer.name
            ].index
            _apply_layer_color(selected_layer, colors[layer_indices])

            # store latest cluster indeces in the features table
            if self.hue_axis == "MANUAL_CLUSTER_ID":
                selected_layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                    color_indices[layer_indices]).astype("category")

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

    color_mapping = {
        napari.layers.Points: lambda _layer, _color: setattr(
            _layer, "face_color", _color
        ),
        napari.layers.Vectors: lambda _layer, _color: setattr(
            _layer, "edge_color", _color
        ),
        napari.layers.Surface: lambda _layer, _color: setattr(
            _layer, "vertex_colors", _color
        ),
        napari.layers.Shapes: lambda _layer, _color: setattr(
            _layer, "face_color", _color
        ),
        napari.layers.Labels: lambda _layer, _color: setattr(
            _layer,
            "colormap",
            DirectLabelColormap(
                color_dict={
                    label: _color[label] for label in np.unique(_layer.data)
                }
            ),
        ),
    }

    if type(layer) in color_mapping:
        if type(layer) is napari.layers.Labels:
            # add a color for the background at the first index
            colors = np.insert(colors, 0, [0, 0, 0, 0], axis=0)
        color_mapping[type(layer)](layer, colors)
        layer.refresh()
