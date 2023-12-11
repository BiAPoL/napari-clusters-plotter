from pathlib import Path

import numpy as np
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox, make_cat10_mod_cmap
from napari.layers import Labels, Points, Tracks
from napari_matplotlib.base import SingleAxesWidget
from napari_matplotlib.util import Interval
from qtpy import uic
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

# icon_folder_path = Path().parent.resolve().parent / 'icons' # Use this line if inside a juptyer notebook
icon_folder_path = (
    Path(__file__).parent / "icons"
)  # Use this line if inside a python script


class PlotterWidget(SingleAxesWidget):
    # Amount of available input layers
    n_layers_input = Interval(1, None)
    # All layers that have a .features attributes
    input_layer_types = (Labels, Points, Tracks)

    def __init__(self, napari_viewer, parent=None):
        super().__init__(napari_viewer, parent=parent)

        self.control_widget = QWidget()
        uic.loadUi(Path(__file__).parent / "plotter_controls.ui", self.control_widget)

        # Add selection tools layout below canvas
        self.selection_tools_layout = self._build_selection_toolbar_layout()

        # Add buttons to selection_toolbar
        self.selection_toolbar.add_custom_button(
            name="Lasso Selection",
            tooltip="Click to enable/disable Lasso selection",
            default_icon_path=icon_folder_path / "button1.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "button1_checked.png",
        )
        # Connect button to callback
        self.selection_toolbar.connect_button_callback(
            name="Lasso Selection", callback=self.on_enable_lasso_selector
        )

        # Set selection colormap
        self.colormap = make_cat10_mod_cmap(first_color_transparent=False)
        # Create instance of CustomScatter
        self.scatter_plot = CustomScatter(self.axes, self.colormap)
        # Add lasso selector
        self.scatter_plot.add_lasso_selector()

        # Add selection tools layout to main layout below matplotlib toolbar and above canvas
        self.layout().insertLayout(2, self.selection_tools_layout)

        self.layout().addWidget(self.control_widget)

    def _build_selection_toolbar_layout(self):
        # Add selection tools layout below canvas
        selection_tools_layout = QHBoxLayout()
        # Add selection toolbar
        self.selection_toolbar = CustomToolbarWidget(self)
        selection_tools_layout.addWidget(self.selection_toolbar)
        # Add cluster spinbox
        selection_tools_layout.addWidget(QLabel("Cluster:"))
        self.cluster_spinbox = QtColorSpinBox(first_color_transparent=False)
        selection_tools_layout.addWidget(self.cluster_spinbox)
        # Add stretch to the right to push buttons to the left
        selection_tools_layout.addStretch(1)
        return selection_tools_layout

    def on_enable_lasso_selector(self, checked):
        if checked:
            print("Lasso selection enabled")
            self.scatter_plot.lasso_selector.enable()
        else:
            print("Lasso selection disabled")
            self.scatter_plot.lasso_selector.disable()


class CustomScatter:
    def __init__(self, axes, colormap, initial_size=50):
        self._axes = axes
        self._colormap = colormap
        self._scatter_handle = self._axes.scatter([], [], s=initial_size, c="none")
        self._current_colors = None
        self._color_indices = None
        self._selected_color_index = 0

    def update_scatter(self, x_data=None, y_data=None):
        if x_data is not None and y_data is not None:
            # self._scatter_handle.set_offsets(np.column_stack([x_data, y_data]))
            self._scatter_handle = self._axes.scatter(x_data, y_data)
            self._update_axes_limits_with_margin(x_data, y_data)
        # Initialize colors if not already done
        if self._current_colors is None:
            # Set color indices with color index 0
            self.color_indices = 1  # temporary value for testing!!

    def _update_axes_limits_with_margin(self, x_data, y_data):
        x_range = max(x_data) - min(x_data)
        y_range = max(y_data) - min(y_data)
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05
        self._axes.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
        self._axes.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        self._axes.relim()  # Recalculate the data limits
        self._axes.autoscale_view()  # Auto-adjust the axes limits
        self._axes.figure.canvas.draw_idle()

    @property
    def data(self):
        return self._scatter_handle.get_offsets()

    @data.setter
    def data(self, xy):
        x_data, y_data = xy
        self.update_scatter(x_data, y_data)

    @property
    def selected_color_index(self):
        return self._selected_color_index

    @selected_color_index.setter
    def selected_color_index(self, index):
        self._selected_color_index = index

    @property
    def colors(self):
        return self._scatter_handle.get_facecolor()

    @colors.setter
    def colors(self, new_colors):
        # Store alpha values
        alpha = self.alphas
        self._current_colors = new_colors
        self._scatter_handle.set_facecolor(self._current_colors)
        if alpha is not None:
            self.alphas = alpha  # Restore alpha values
        self._axes.figure.canvas.draw_idle()  # maybe unecessary because alpha updates the canvas

    @property
    def color_indices(self):
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices):
        # Do nothing if there is no data
        if len(self.data) == 0:
            return
        # Handle scalar indices
        if np.isscalar(indices):
            indices = np.full(self.data.shape[0], indices)
        self._color_indices = indices
        new_colors = self._colormap(indices)
        # update scatter colors
        self.colors = new_colors

    @property
    def alphas(self):
        if self._current_colors is not None:
            return self._current_colors[:, -1]
        return None

    @alphas.setter
    def alphas(self, alpha_values):
        if self._current_colors is not None:
            # Handle scalar alpha value
            if np.isscalar(alpha_values):
                alpha_values = np.full(self._current_colors.shape[0], alpha_values)
            self._current_colors[:, -1] = alpha_values  # Update alpha values
            self._scatter_handle.set_facecolor(self._current_colors)
            self._axes.figure.canvas.draw_idle()

    def add_lasso_selector(self):
        self.lasso_selector = CustomLassoSelector(self, self._axes)


class CustomLassoSelector:
    def __init__(self, parent, axes):
        self.artist = parent
        self.axes = axes
        self.canvas = axes.figure.canvas

        self.lasso = LassoSelector(axes, onselect=self.onselect)
        self.ind = []
        self.ind_mask = []
        # start disabled
        self.disable()

    def enable(self):
        """Enable the Lasso selector."""
        self.lasso = LassoSelector(self.axes, onselect=self.onselect)

    def disable(self):
        """Disable the Lasso selector."""
        self.lasso.disconnect_events()

    def onselect(self, verts):
        # Get plotted data and color indices
        plotted_data = self.artist.data
        color_indices = self.artist.color_indices
        # Get indices of selected data points
        path = mplPath(verts)
        self.ind_mask = path.contains_points(plotted_data)
        self.ind = np.nonzero(self.ind_mask)[0]
        # Set selected indices with selected color index
        color_indices[self.ind] = self.artist.selected_color_index
        # TODO: Replace this by pyq signal/slot
        self.artist.color_indices = color_indices  # This updates the plot
