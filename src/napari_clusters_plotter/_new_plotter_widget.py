from enum import Enum, auto
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from biaplotter.plotter import ArtistType, CanvasWidget
from napari.utils.colormaps import ALL_COLORMAPS
from qtpy import uic
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QMainWindow, QScrollArea, QVBoxLayout,
                            QWidget)

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

    input_layer_types = [
        napari.layers.Labels,
        napari.layers.Points,
        napari.layers.Surface,
        napari.layers.Vectors,
    ]

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

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

        self._update_layers(None)
        self._setup_callbacks()

    def _setup_callbacks(self):

        # Adding Connections
        self.control_widget.plot_type_box.currentIndexChanged.connect(
            self._plotting_type_changed
        )
        self.control_widget.set_bins_button.clicked.connect(
            self._bin_number_set
        )
        self.control_widget.auto_bins_checkbox.stateChanged.connect(
            self._bin_auto
        )
        self.control_widget.log_scale_checkbutton.stateChanged.connect(
            self._replot
        )
        self.control_widget.non_selected_checkbutton.stateChanged.connect(
            self._checkbox_status_changed
        )
        self.control_widget.cmap_box.currentIndexChanged.connect(self._replot)

        self.viewer.layers.selection.events.changed.connect(
            self._update_layers
        )

        # reset the coloring of the selected layer
        self.control_widget.reset_button.clicked.connect(self._reset)

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].currentTextChanged.connect(self._replot)

        # connect data selection in plot to layer coloring update
        self.plotting_widget.active_artist.color_indices_changed_signal.connect(
            self._add_manual_cluster_id
        )

    def _replot(self):

        # if no x or y axis is selected, return
        if self.x_axis == "" or self.y_axis == "":
            return

        data_to_plot = self._get_data()
        self.plotting_widget.active_artist.data = data_to_plot
        # redraw the whole plot
        try:
            # plotting function needs to be here
            pass

        except AttributeError:
            # In this case, replotting is not yet possible
            pass

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
    def plotting_type(self, type):
        if type in PlottingType.__members__.keys():
            self.control_widget.plot_type_box.setCurrentText(type)

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
        self.control_widget.hue_box.setCurrentText(
            column
        )  # TODO insert checks and change values

    @property
    def n_selected_layers(self) -> int:
        """
        Number of currently selected layers.
        """
        return len(list(self.viewer.layers.selection))

    def _get_data(self) -> np.ndarray:
        """
        Get the data from the selected layers features.
        """
        features = self._get_features()
        x_data = features[self.x_axis].values
        y_data = features[self.y_axis].values

        # if no hue is selected, set it to 0
        if self.hue_axis == "None":
            hue = np.zeros(len(features))
        elif self.hue_axis != "":
            hue = features[self.hue_axis].values

        return np.stack([x_data, y_data], axis=1)


    def _update_layers(self, event: napari.utils.events.Event) -> None:
        """
        Update the layers list when the selection changes.
        """
        # don't do anything if no layer is selected
        if self.n_selected_layers == 0:
            return

        self._update_feature_selection(None)

        for layer in list(self.viewer.layers.selection):
            layer.events.features.connect(self._update_feature_selection)

    def _update_feature_selection(
        self, event: napari.utils.events.Event
    ) -> None:
        """
        Update the features in the dropdowns.
        """
        # block selector changed signals until all items added
        for dim in ["x", "y", "hue"]:
            self._selectors[dim].blockSignals(True)

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].clear()

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].addItems(self.common_columns)

        # it should always be possible to select no color
        self._selectors["hue"].addItem("None")

        for dim in ["x", "y", "hue"]:
            self._selectors[dim].blockSignals(False)

        features = self._get_features()
        if self.n_selected_layers > 0 and not features.empty:
            self.x_axis = self.common_columns[0]
            self.y_axis = self.common_columns[0]

    def _add_manual_cluster_id(self):
        """
        Color the selected layer according to the color indices.
        """

        features = self._get_features()
        for selected_layer in list(self.viewer.layers.selection):

            # turn the color indices into an array of RGBA colors
            color_indeces = self.plotting_widget.active_artist.color_indices
            color = self.plotting_widget.active_artist.categorical_colormap(
                color_indeces
            )

            # pull the correct rows from the features dataframe that correspond to the
            # selected layer and use to identify the correct colors
            indeces = features[features["layer"] == selected_layer.name].index
            _color_layer(selected_layer, color[indeces])
            selected_layer.refresh()

    def _reset(self):
        """
        Reset the selection in the current plotting widget.
        """
        self.plotting_widget.active_artist.color_indices = np.zeros(
            len(self._get_features())
        )
        self._add_manual_cluster_id()


def _color_layer(layer, color):
    """
    Color the layer according to the color array. This needs to be done in a different
    way for each layer type.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to color.

    color : np.ndarray
        The color array (Nx4).
    """
    from napari.utils import DirectLabelColormap

    if isinstance(layer, napari.layers.Points):
        layer.face_color = color
    elif isinstance(layer, napari.layers.Vectors):
        layer.edge_color = color
    elif isinstance(layer, napari.layers.Surface):
        layer.vertex_colors = color
    elif isinstance(layer, napari.layers.Labels):
        color_dict = {}
        for label in np.unique(layer.data):
            color_dict[label] = color[label]
        color_dict[0] = [0, 0, 0, 0]  # make sure background is transparent
        colormap = DirectLabelColormap(color_dict=color_dict)
        layer.colormap = colormap
    layer.refresh()
