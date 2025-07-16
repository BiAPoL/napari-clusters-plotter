from enum import Enum, auto
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from biaplotter.artists import Histogram2D, Scatter
from biaplotter.colormap import BiaColormap
from biaplotter.plotter import CanvasWidget
from matplotlib.cm import viridis
from matplotlib.colors import LinearSegmentedColormap
from nap_plot_tools.cmap import (
    cat10_mod_cmap,
    cat10_mod_cmap_first_transparent,
)
from napari.utils.colormaps import ALL_COLORMAPS
from napari.utils.notifications import show_info, show_warning
from qtpy import uic
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ._algorithm_widget import BaseWidget


class PlottingType(Enum):
    HISTOGRAM2D = auto()
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
        self.layers_being_unselected = []
        self._on_update_layer_selection(None)
        self._setup_callbacks()

        self.plot_needs_update.connect(self._replot)

        # Colormap reference to be indexed like this:
        # reference[is_categorical, plot_type]
        self.colormap_reference = {
            (True, "HISTOGRAM2D"): cat10_mod_cmap_first_transparent,
            (True, "SCATTER"): cat10_mod_cmap,
            (False, "HISTOGRAM2D"): self._napari_to_mpl_cmap(
                self.overlay_colormap_plot
            ),
            (False, "SCATTER"): self._napari_to_mpl_cmap(
                self.overlay_colormap_plot
            ),
        }
        self.plot_needs_update.emit()

    def _napari_to_mpl_cmap(self, colormap_name):
        return LinearSegmentedColormap.from_list(
            ALL_COLORMAPS[colormap_name].name,
            ALL_COLORMAPS[colormap_name].colors,
        )

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
        self.plotting_widget.artists["HISTOGRAM2D"]._histogram_colormap = (
            BiaColormap(viridis)
        )  # Start histogram colormap with viridis
        self.plotting_widget.active_artist = "SCATTER"

        # Add plot and options as widgets
        self.layout.addWidget(self.plotting_widget)
        self.layout.addWidget(self.control_widget)

        # Setting of Widget options
        self.hue: QComboBox = self.control_widget.hue_box

        self.control_widget.plot_type_box.addItems(["SCATTER", "HISTOGRAM2D"])
        # Fill overlay colormap box with all available colormaps
        self.control_widget.overlay_cmap_box.addItems(
            list(ALL_COLORMAPS.keys())
        )
        self.control_widget.overlay_cmap_box.setCurrentIndex(
            np.argwhere(np.array(list(ALL_COLORMAPS.keys())) == "magma")[0][0]
        )
        # Fill histogram colormap box with all available colormaps
        self.control_widget.histogram_cmap_box.addItems(
            list(ALL_COLORMAPS.keys())
        )
        self.control_widget.histogram_cmap_box.setCurrentIndex(
            np.argwhere(np.array(list(ALL_COLORMAPS.keys())) == "viridis")[0][
                0
            ]
        )

        # Setting Visibility Defaults
        self.control_widget.cmap_container.setVisible(False)
        self.control_widget.bins_settings_container.setVisible(False)
        self.control_widget.additional_options_container.setVisible(False)

    def contextMenuEvent(self, event):
        self.context_menu.exec_(event.globalPos())

    def _on_export_clusters(self):
        """
        Export the selected cluster to a new layer.
        """

        # get currently selected cluster from plotting widget
        selected_cluster = self.plotting_widget.class_spinbox.value
        features = self._get_features()
        hue_column = self.hue_axis
        if hue_column not in self.categorical_columns:
            show_warning(
                '"Selected hue axis is not categorical, cannot export clusters.'
            )
            return

        # get the layer to export from
        for layer in self.layers:
            features_subset = features[
                features["layer"] == layer.unique_id
            ].reset_index()
            indices = features_subset[hue_column].values == selected_cluster
            if not np.any(indices):
                show_info(
                    "No data points found for selected cluster"
                    f"{selected_cluster} in layer {layer.name}."
                )
                continue
            export_layer = _export_cluster_to_layer(
                layer, indices, subcluster_index=selected_cluster
            )
            if export_layer is not None:
                self.viewer.add_layer(export_layer)

    def _setup_callbacks(self):
        """
        Set up the callbacks for the widget.
        """

        # Connect all necessary functions to the replot
        connections_to_replot = [
            (
                self.control_widget.log_scale_checkbutton.toggled,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.histogram_cmap_box.currentTextChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.n_bins_box.valueChanged,
                self.plot_needs_update.emit,
            ),
            (
                self.control_widget.non_selected_checkbutton.stateChanged,
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
        self.plotting_widget.show_color_overlay_signal.connect(
            self._on_show_plot_overlay
        )

        # connect scatter/histogram switch
        self.control_widget.plot_type_box.currentTextChanged.connect(
            self._on_plot_type_changed
        )
        self.control_widget.overlay_cmap_box.currentTextChanged.connect(
            self._on_overlay_colormap_changed
        )
        self.control_widget.auto_bins_checkbox.toggled.connect(
            self._on_bin_auto_toggled
        )

        self.control_widget.pushButton_export_layer.clicked.connect(
            self._on_export_clusters
        )

    def _on_finish_draw(self, color_indices: np.ndarray):
        """
        Called when user finsihes drawing. Will change the hue combo box to the
        feature 'MANUAL_CLUSTER_ID', which then triggers a redraw.
        """

        # if the hue axis is not set to MANUAL_CLUSTER_ID, set it to that
        # otherwise replot the data

        if self.n_selected_layers == 0:
            return

        features = self._get_features()
        for layer in self.layers:
            layer_indices = features[
                features["layer"] == layer.unique_id
            ].index

            # store latest cluster indeces in the features table
            layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                color_indices[layer_indices]
            ).astype("category")

        if self.hue_axis != "MANUAL_CLUSTER_ID":
            self.hue_axis = "MANUAL_CLUSTER_ID"

        self.plot_needs_update.emit()

    def _handle_advanced_options_widget_visibility(self):
        """
        Control visibility of overlay colormap box and log scale checkbox
        based on the selected hue axis and active artist.
        """
        active_artist = self.plotting_widget.active_artist
        # Control visibility of overlay colormap box and log scale checkbox
        if self.hue_axis in self.categorical_columns:
            self.control_widget.overlay_cmap_box.setEnabled(False)
            self.control_widget.log_scale_checkbutton.setEnabled(False)
            if isinstance(active_artist, Histogram2D):
                # Enable if histogram to allow log scale of histogram itself
                self.control_widget.log_scale_checkbutton.setEnabled(True)
        else:
            self.control_widget.overlay_cmap_box.setEnabled(True)
            self.control_widget.log_scale_checkbutton.setEnabled(True)

        if isinstance(active_artist, Histogram2D):
            self.control_widget.cmap_container.setVisible(True)
            self.control_widget.bins_settings_container.setVisible(True)
        else:
            self.control_widget.cmap_container.setVisible(False)
            self.control_widget.bins_settings_container.setVisible(False)

    def _reset_axes_labels(self):
        """
        Clear the x and y axis labels in the plotting widget.
        """
        for artist in self.plotting_widget.artists.values():
            if hasattr(artist, "x_label"):
                artist.x_label_text = ""
                artist.x_label_color = "white"
            if hasattr(artist, "y_label"):
                artist.y_label_text = ""
                artist.y_label_color = "white"

    def _replot(self):
        """
        Replot the data with the current settings.
        """
        # check if there are any valid layers selected
        if len(self.layers) == 0:
            self._clean_up()
            return

        # if no x or y axis is selected, return
        if self.x_axis == "" or self.y_axis == "":
            return

        # retrieve the data from the selected layers
        features = self._get_features()
        x_data = features[self.x_axis].values
        y_data = features[self.y_axis].values

        # select appropriate overlay colormap for usecase
        overlay_cmap = self.colormap_reference[
            (self.hue_axis in self.categorical_columns, self.plotting_type)
        ]
        self._handle_advanced_options_widget_visibility()
        self._reset_axes_labels()
        active_artist = self.plotting_widget.active_artist
        active_artist.x_label_text = self.x_axis
        active_artist.y_label_text = self.y_axis
        color_norm = "log" if self.log_scale else "linear"
        # First set the data related properties in the active artist
        active_artist.data = np.stack([x_data, y_data], axis=1)
        if isinstance(active_artist, Histogram2D):
            active_artist.histogram_colormap = self._napari_to_mpl_cmap(
                self.histogram_colormap_plot
            )
            if self.automatic_bins:
                number_bins = int(
                    np.max(
                        [
                            self._estimate_number_bins(x_data),
                            self._estimate_number_bins(y_data),
                        ]
                    )
                )
                # Block signal to avoid replotting while setting value
                self.control_widget.n_bins_box.blockSignals(True)
                self.bin_number = number_bins
                self.control_widget.n_bins_box.blockSignals(False)
            active_artist.bins = self.bin_number
            active_artist.histogram_color_normalization_method = color_norm

        # Then set color_indices and colormap properties in the active artist
        active_artist.overlay_colormap = overlay_cmap
        active_artist.color_indices = features[self.hue_axis].to_numpy()

        # Force overlay to be visible if non-categorical hue axis is selected
        if self.hue_axis not in self.categorical_columns:
            self.plotting_widget.show_color_overlay = True

        # If color_indices are all zeros (no selection) and the hue axis
        # is categorical, apply default colors
        if (
            np.all(active_artist.color_indices == 0)
            and self.hue_axis in self.categorical_columns
        ):
            self._update_layer_colors(use_color_indices=False)

        # Otherwise, color layer by value (optionally applying log scale)
        else:
            if isinstance(active_artist, Histogram2D):
                active_artist.overlay_color_normalization_method = color_norm
            elif isinstance(active_artist, Scatter):
                active_artist.color_normalization_method = color_norm
            self._update_layer_colors(use_color_indices=True)

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

    def _on_plot_type_changed(self):
        """
        Called when the plot type changes.
        """
        if self.plotting_type == PlottingType.HISTOGRAM2D.name:
            self.plotting_widget.active_artist = "HISTOGRAM2D"
            self.plotting_widget.active_artist.overlay_colormap = (
                cat10_mod_cmap_first_transparent
            )

        elif self.plotting_type == PlottingType.SCATTER.name:
            self.plotting_widget.active_artist = "SCATTER"
            self.plotting_widget.active_artist.overlay_colormap = (
                cat10_mod_cmap
            )
        self.plot_needs_update.emit()

    def _on_overlay_colormap_changed(self):
        colormap_name = self.overlay_colormap_plot
        # Dynamically update the colormap_reference dictionary
        self.colormap_reference[(False, "HISTOGRAM2D")] = (
            self._napari_to_mpl_cmap(colormap_name)
        )
        self.colormap_reference[(False, "SCATTER")] = self._napari_to_mpl_cmap(
            colormap_name
        )
        self.plot_needs_update.emit()

    def _on_histogram_colormap_changed(self):
        self.plot_needs_update.emit()

    def _checkbox_status_changed(self):
        self.plot_needs_update.emit()

    def _on_bin_auto_toggled(self, state: bool):
        """
        Called when the automatic bin checkbox is toggled.
        Enables or disables the bin number box accordingly.
        """
        self.control_widget.n_bins_box.setEnabled(not state)
        self.plot_needs_update.emit()

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

    @bin_number.setter
    def bin_number(self, val: int):
        self.control_widget.n_bins_box.setValue(val)

    @property
    def hide_non_selected(self):
        return self.control_widget.non_selected_checkbutton.isChecked()

    @hide_non_selected.setter
    def hide_non_selected(self, val: bool):
        self.control_widget.non_selected_checkbutton.setChecked(val)

    @property
    def overlay_colormap_plot(self):
        return self.control_widget.overlay_cmap_box.currentText()

    @property
    def histogram_colormap_plot(self):
        return self.control_widget.histogram_cmap_box.currentText()

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
        self.plot_needs_update.emit()

    @property
    def y_axis(self):
        return self.control_widget.y_axis_box.currentText()

    @y_axis.setter
    def y_axis(self, column: str):
        self.control_widget.y_axis_box.setCurrentText(column)
        self.plot_needs_update.emit()

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

    def _estimate_number_bins(self, data) -> int:
        """
        Estimates number of bins according Freedman–Diaconis rule

        Parameters
        ----------
        data: Numpy array

        Returns
        -------
        Estimated number of bins
        """
        from scipy.stats import iqr

        est_a = (np.max(data) - np.min(data)) / (
            2 * iqr(data) / np.cbrt(len(data))
        )
        if np.isnan(est_a):
            return 256
        return int(est_a)

    def _on_update_layer_selection(
        self, event: napari.utils.events.Event
    ) -> None:
        """
        Called when the layer selection changes. Updates the layers attribute.
        """
        # check if the selected layers are of the correct type
        self.layers = self.get_valid_layers()

        # don't do anything if no layer is selected
        if len(self.layers) == 0:
            self._clean_up()
            return

        # insert 'MANUAL_CLUSTER_ID' column if it doesn't exist
        for layer in self.layers:
            if "MANUAL_CLUSTER_ID" not in layer.features.columns:
                layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                    np.zeros(len(layer.features), dtype=np.int32)
                ).astype("category")

        if event is not None and len(event.removed) > 0:
            # remove the layers that are not in the selection anymore
            self.layers_being_unselected = list(event.removed)
        self._update_feature_selection(None)

        for layer in self.layers:
            event_attr = getattr(layer.events, "features", None) or getattr(
                layer.events, "properties", None
            )
            if event_attr:
                event_attr.connect(self._update_feature_selection)
            else:
                show_warning(
                    f"Layer {layer.name} does not have events.features or events.properties"
                )

    def _clean_up(self):
        """In case of empty layer selection"""

        # disconnect the events from the layers
        for layer in self.layers:
            event_attr = getattr(layer.events, "features", None) or getattr(
                layer.events, "properties", None
            )
            if event_attr:
                event_attr.disconnect(self._update_feature_selection)
            else:
                show_warning(
                    f"Layer {layer.name} does not have events.features or events.properties"
                )

        # reset the selected layers
        self.layers = []

        # reset the selectors
        for dim in ["x", "y", "hue"]:
            selector = self._selectors[dim]
            selector.blockSignals(True)
            selector.clear()
            selector.blockSignals(False)

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

    def _on_show_plot_overlay(self, state: bool) -> None:
        """
        Called when the plot overlay is hidden or shown.
        """
        self._update_layer_colors(use_color_indices=state)

    def _generate_default_colors(self, layer):
        """
        Generate default colors for a given layer based on its type.

        Parameters
        ----------
        layer : napari.layers.Layer
            The layer for which to generate default colors.

        Returns
        -------
        np.ndarray
            An array of default colors (Nx4).
        """
        if isinstance(layer, napari.layers.Labels):
            # Use CyclicLabelColormap with N colors
            from ._utilities import _get_unique_values

            # check if is dask or numpy
            n_labels = _get_unique_values(layer).size - 1
            if n_labels >= 2**16:
                np.random.seed(42)  # For reproducibility
                rgba = np.random.uniform(
                    low=0,
                    high=n_labels,
                    size=(n_labels, 4),
                )
                rgba[:, 3] = 1.0  # Set alpha to 1 for all colors
            else:
                from napari.utils.colormaps.colormap_utils import (
                    label_colormap,
                )

                rgba = np.asarray(label_colormap(n_labels).dict()["colors"])
            return rgba
        else:
            # Default to white for other layer types
            default_color = np.array([[1, 1, 1, 1]])
            return default_color.repeat(len(layer.features), axis=0)

    def _update_layer_colors(self, use_color_indices: bool = False) -> None:
        """
        Update colors for the selected layers based on the context.

        Parameters
        ----------
        use_color_indices : bool, optional
            If True, apply colors based on the active artist's color indices
            (unless show_color_overlay is False).
            If False, apply default colors to the layers.
            Defaults to False.
        """
        if self.n_selected_layers == 0:
            return

        # Disable coloring based on color_indices if overlay toggle unchecked
        if not self.plotting_widget.show_color_overlay:
            use_color_indices = False

        features = self._get_features()
        active_artist = self.plotting_widget.active_artist

        for selected_layer in self.layers:
            if use_color_indices:
                # Apply colors based on color indices
                rgba_colors = active_artist.color_indices_to_rgba(
                    active_artist.color_indices
                )
                layer_indices = features[
                    features["layer"] == selected_layer.unique_id
                ].index
                self._set_layer_color(
                    selected_layer, rgba_colors[layer_indices]
                )

                # Update MANUAL_CLUSTER_ID if applicable
                if self.hue_axis == "MANUAL_CLUSTER_ID":
                    selected_layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                        active_artist.color_indices[layer_indices]
                    ).astype("category")
            else:
                # Apply default colors
                rgba_colors = self._generate_default_colors(selected_layer)
                self._set_layer_color(selected_layer, rgba_colors)

        # Apply default colors to layers being unselected
        for layer in self.layers_being_unselected:
            if layer in self.viewer.layers and self._is_supported_layer(layer):
                rgba_colors = self._generate_default_colors(layer)
                self._set_layer_color(layer, rgba_colors)
        self.layers_being_unselected = []

    def _set_layer_color(self, layer, colors):
        """
        Set colors for a specific layer based on its type.

        Parameters
        ----------
        layer : napari.layers.Layer
            The layer to color.

        colors : np.ndarray
            The color array (Nx4).
        """
        if isinstance(layer, napari.layers.Points):
            layer.face_color = colors
        elif isinstance(layer, napari.layers.Vectors):
            layer.edge_color = colors
        elif isinstance(layer, napari.layers.Surface):
            layer.vertex_colors = colors
        elif isinstance(layer, napari.layers.Shapes):
            layer.edge_color = colors
        elif isinstance(layer, napari.layers.Tracks):
            layer._track_colors = colors
            layer.events.color_by()
        elif isinstance(layer, napari.layers.Labels):
            from napari.utils import DirectLabelColormap

            from ._utilities import _get_unique_values

            # Ensure the first color is transparent for the background
            colors = np.insert(colors, 0, [0, 0, 0, 0], axis=0)
            color_dict = dict(zip(_get_unique_values(layer), colors))
            layer.colormap = DirectLabelColormap(color_dict=color_dict)
        layer.refresh()

    def _reset(self):
        """
        Reset the selection in the current plotting widget.
        """
        if self.n_selected_layers == 0:
            return

        for layer in self.layers:
            if "MANUAL_CLUSTER_ID" in layer.features.columns:
                layer.features["MANUAL_CLUSTER_ID"] = pd.Series(
                    np.zeros(len(layer.features), dtype=np.int32)
                ).astype("category")
        # self.plotting_widget.active_artist.color_indices = np.zeros(
        #     len(self._get_features())
        # )
        self._update_layer_colors(use_color_indices=False)
        self.control_widget.hue_box.setCurrentText("MANUAL_CLUSTER_ID")
        self.plot_needs_update.emit()


def _export_cluster_to_layer(
    layer: "napari.layers.Layer",
    export_indices: np.ndarray,
    subcluster_index: int = None,
) -> "napari.layers.Layer":
    """
    Export the selected cluster to a new layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to export the cluster from.

    export_indices : np.ndarray
        The indices of the cluster to export.

    subcluster_index : str
        The name of the new layer. If not provided, the name of the layer will be used.

    Returns
    -------
    napari.layers.Layer
        The new layer with the selected cluster.
    """
    new_features = layer.features.iloc[export_indices].copy()

    if isinstance(layer, napari.layers.Labels):
        from skimage.segmentation import relabel_sequential

        LUT = np.arange(layer.data.max() + 1)
        LUT[1:][~export_indices] = 0
        new_data = LUT[layer.data]
        new_data, forward_map, _ = relabel_sequential(
            new_data,
        )
        new_features["original_label"] = pd.Categorical(
            forward_map.in_values[1:]
        )
        new_features["label"] = pd.Categorical(forward_map.out_values[1:])
        new_layer = napari.layers.Labels(new_data)

    elif isinstance(layer, napari.layers.Points):
        new_layer = napari.layers.Points(layer.data[export_indices])
        new_layer.size = layer.size[export_indices]

    elif isinstance(layer, napari.layers.Shapes):
        new_shapes = [
            shape for shape, i in zip(layer.data, export_indices) if i
        ]
        new_shape_types = np.asarray(layer.shape_type)[export_indices]
        new_layer = napari.layers.Shapes(
            new_shapes, shape_type=new_shape_types
        )
        edge_widths = list(np.asarray(layer.edge_width)[export_indices])
        new_layer.edge_width = edge_widths

    elif isinstance(layer, napari.layers.Tracks):
        new_tracks = layer.data[export_indices]
        new_layer = napari.layers.Tracks(new_tracks)

    elif isinstance(layer, napari.layers.Surface):
        new_vertices = layer.data[0][export_indices]
        old_to_new_index = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(np.where(export_indices)[0])
        }

        # Vectorized update of faces using list comprehension
        new_faces = [
            [old_to_new_index[vertex_idx] for vertex_idx in face]
            for face in layer.data[1]
            if all(vertex_idx in old_to_new_index for vertex_idx in face)
        ]
        new_layer = napari.layers.Surface(
            (new_vertices, np.asarray(new_faces, dtype=np.int32)),
        )

    elif isinstance(layer, napari.layers.Vectors):
        new_layer = napari.layers.Vectors(layer.data[export_indices])

    else:
        return None

    new_layer.scale = layer.scale
    new_layer.translate = layer.translate
    new_layer.rotate = layer.rotate

    if not subcluster_index:
        new_layer.name = f"{layer.name} subcluster"
    else:
        new_layer.name = f"{layer.name} subcluster {subcluster_index}"

    # copy features to new layer if available and drop cluster column
    new_layer.features = new_features

    return new_layer
