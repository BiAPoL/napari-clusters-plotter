import numpy as np
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector


class Selector:
    """
    Base class for a selector.

    Parameters
    ----------
    parent : object
        The parent object.
    axes : matplotlib.axes.Axes
        The axes to draw on.
    """

    def __init__(self, parent, axes):
        self.parent = parent
        self.axes = axes
        self.canvas = self.axes.figure.canvas
        self.selector = None

        self.indeces = []
        self.indeces_mask = []

        self.disable()

    def on_select(self, verts):
        self._on_select(verts)

    def on_drag(self, event):
        self._on_drag(event)

    def on_release(self, event):
        self._on_release(event)

    def enable(self):
        """Enable the selector."""
        self._enable()

    def disable(self):
        self.selector.disconnect_events()

    def _enable(self):
        raise NotImplementedError

    def _on_select(self, verts):
        raise NotImplementedError

    def _on_drag(self, event):
        raise NotImplementedError

    def _on_release(self, event):
        raise NotImplementedError


class CustomLassoSelector(Selector):
    """
    Lasso selector.

    Parameters
    ----------
    parent : object
        The parent object.
    axes : matplotlib.axes.Axes
        The axes to draw on.
    """

    def __init__(self, parent, axes):
        super().__init__(parent, axes)

    def _enable(self):
        self.selector = LassoSelector(self.axes, onselect=self.on_select)

    def _on_select(self, verts):
        # Get plotted data and color indices
        plotted_data = self.artist.data
        color_indices = self.artist.color_indices
        # Get indices of selected data points
        path = mplPath(verts)
        self.indeces_mask = path.contains_points(plotted_data)
        self.indeces = np.nonzero(self.indeces_mask)[0]
        # Set selected indices with selected color index
        color_indices[self.indeces] = self.artist.selected_color_index
        # TODO: Replace this by pyq signal/slot
        self.artist.color_indices = color_indices  # This updates the plot
