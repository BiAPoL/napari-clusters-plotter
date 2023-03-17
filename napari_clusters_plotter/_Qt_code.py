import os
import typing
from pathlib import Path as PathL

import numpy as np
import numpy.typing
import pandas as pd
from magicgui.widgets import create_widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from napari.layers import Image, Labels
from qtpy.QtCore import QRect
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

ICON_ROOT = PathL(__file__).parent / "icons"
MAX_WIDTH = 100


def collapsible_box(name):
    return QCollapsible(name)


def measurements_container_and_list():
    """
    Creates a container widget and a list widget for displaying measurements.

    Returns
    -------
    A tuple containing the created container widget and the list widget for displaying measurements.
    """
    properties_container = QWidget()
    properties_container.setLayout(QVBoxLayout())
    properties_container.layout().addWidget(QLabel("Measurements"))
    properties_list = QListWidget()
    properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    properties_list.setGeometry(QRect(10, 10, 101, 291))
    properties_container.layout().addWidget(properties_list)

    return properties_container, properties_list


def labels_container_and_selection():
    """
    Create a container and a dropdown widget to select the labels layer.

    Returns
    -------
    A tuple containing a QWidget for displaying the labels layer selection container,
    and a QWidget containing the selection options for the labels layer.
    """
    labels_layer_selection_container = QWidget()
    labels_layer_selection_container.setLayout(QHBoxLayout())
    labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
    labels_select = create_widget(annotation=Labels, label="labels_layer")
    labels_layer_selection_container.layout().addWidget(labels_select.native)

    return labels_layer_selection_container, labels_select


def image_container_and_selection():
    """
    Creates a container for selecting an image layer.

    Returns
    -------
    A tuple containing a QWidget for displaying the image layer selection container,
    and a QWidget containing the selection options for the image layer.
    """
    image_select = create_widget(annotation=Image, label="image_layer")
    image_layer_selection_container = QWidget()
    image_layer_selection_container.setLayout(QHBoxLayout())
    image_layer_selection_container.layout().addWidget(QLabel("Image layer"))
    image_layer_selection_container.layout().addWidget(image_select.native)

    return image_layer_selection_container, image_select


def title(name: str):
    """
    Creates a widget with a single label displaying the specified name as the title.

    Parameters
    ----------
    name : str
        The text to be displayed as the title.

    Returns
    -------
    A widget with a single label displaying the specified name as the title.
    """
    title_container = QWidget()
    title_container.setLayout(QVBoxLayout())
    title_container.layout().addWidget(QLabel(name))

    return title_container


def int_sbox_containter_and_selection(
    name: str,
    value: int,
    min: int = 2,
    label: str = "Number of Clusters",
    visible: bool = False,
    max_width: int = MAX_WIDTH,
    tool_tip: str = None,
    tool_link: str = None,
):
    """
    Creates a container widget for an integer spin box and returns the container widget and the spin box.

    Parameters
    ----------
    name : str
        A unique name for the spin box.
    value : int
        The initial value of the spin box.
    min : int, optional
        The minimum value of the spin box. Default is 2.
    label : str, optional
        The label of the container widget. Default is "Number of Clusters".
    visible : bool, optional
        Whether the container widget is visible. Default is False.
    max_width : int, optional
        The maximum width of the spin box widget. Default is MAX_WIDTH.
    tool_tip : str, optional
        The tooltip to be displayed for the container widget.
    tool_link : str, optional
        The hyperlink URL to associate with the tooltip.

    Returns
    -------
    container : QWidget
        A container widget that holds the spin box widget.
    selection : QSpinBoxWidget
        A spin box widget.
    """
    container = QWidget()
    container.setLayout(QHBoxLayout())
    container.layout().addWidget(QLabel(label))

    selection = create_widget(
        widget_type="SpinBox",
        name=name,
        value=value,
        options={"min": min, "step": 1},
    )

    container.layout().addStretch()
    selection.native.setMaximumWidth(max_width)
    container.layout().addWidget(selection.native)
    container.setVisible(visible)

    if tool_link is not None or tool_tip is not None:
        add_tooltip(container, tool_link, tool_tip)

    return container, selection


def float_sbox_containter_and_selection(
    name: str,
    value: float,
    label: str,
    min: float = 0,
    step: float = 0.1,
    max: float = 1,
    visible: bool = False,
    max_width: int = MAX_WIDTH,
    tool_tip: str = None,
    tool_link: str = None,
):
    """
    Creates a container widget for a float spin box and returns the container widget and the spin box.

    Parameters
    ----------
    name : str
        A unique name for the spin box.
    value : float
        The initial value of the spin box.
    label : str
        The label of the container widget.
    min : float, optional
        The minimum value of the spin box. Default is 0.
    step : float, optional
        The step size of the spin box. Default is 0.1.
    max : float, optional
        The maximum value of the spin box. Default is 1.
    visible : bool, optional
        Whether the container widget is visible. Default is False.
    max_width : int, optional
        The maximum width of the spin box widget. Default is MAX_WIDTH.
    tool_tip : str, optional
        The tooltip to be displayed for the container widget.
    tool_link : str, optional
        The hyperlink URL to associate with the tooltip.

    Returns
    -------
    A tuple consisting of the container widget that holds the spin box widget, and
    a float spin box widget for the selection of the value.
    """
    container = QWidget()
    container.setLayout(QHBoxLayout())
    container.layout().addWidget(QLabel(label))

    selection = create_widget(
        widget_type="FloatSpinBox",
        name=name,
        value=value,
        options={"min": min, "step": step, "max": max},
    )

    container.layout().addStretch()
    selection.native.setMaximumWidth(max_width)
    container.layout().addWidget(selection.native)
    container.setVisible(visible)

    if tool_link is not None or tool_tip is not None:
        add_tooltip(container, tool_link, tool_tip)

    return container, selection


def button(name):
    """
    Creates a container widget for a QPushButton and returns the container widget and the button.

    Parameters
    ----------
    name : str
        The text to display on the button.

    Returns
    -------
    widget : QWidget
        A container widget that holds the button widget.
    button : QPushButton
        A QPushButton widget.
    """
    widget = QWidget()
    widget.setLayout(QHBoxLayout())
    button = QPushButton(name)
    widget.layout().addWidget(button)
    return widget, button


def checkbox(
    name: str,
    value: bool,
    visible: bool = False,
    tool_tip: str = None,
    tool_link: str = None,
):
    container = QWidget()
    container.setLayout(QHBoxLayout())

    if tool_tip is not None or tool_link is not None:
        selection = create_widget(
            widget_type="CheckBox",
            name=name,
            value=value,
            options={"tooltip": tool_tip},
        )
    else:
        selection = create_widget(
            widget_type="CheckBox",
            name=name,
            value=value,
        )
    container.layout().addWidget(selection.native)
    container.setVisible(visible)

    return container, selection


def add_tooltip(
    container,
    tool_link: str,
    tool_tip: str,
):
    """
    Add a tooltip to a widget that displays additional information when hovered or clicked on.

    Parameters
    ----------
    container : QWidget
        The container widget that the tooltip should be added to.
    tool_link : str
        The URL of a webpage that provides more information about the parameter.
    tool_tip : str
        The text to be displayed as the tooltip when hovered over the question mark.
    """
    help_tooltip = QLabel()
    new_line = "\n"
    if tool_link is not None:
        help_tooltip.setOpenExternalLinks(True)
        help_tooltip.setText(
            f'<a href="{tool_link}" '
            'style="text-decoration:none; color:white"><b>?</b></a>'
        )
    if tool_tip is not None:
        help_tooltip.setToolTip(
            f"{tool_tip}{new_line}{'Click on a question mark to read more.' if tool_link is not None else ''}"
        )

    container.layout().addWidget(help_tooltip)


def algorithm_choice(name: str, value, options: dict, label: str):
    """
    Create a widget for selecting a clustering algorithm from a set of options.

    Parameters
    ----------
    name : str
        The name to be used for the widget.
    value :
    The initial value of the widget.
    options : dict
        A dictionary of possible options, where the keys are option
        names and the values are corresponding strings that are
        actually displayed in the combobox.
    label : str
        The label to be displayed next to the widget.

    Returns
    ----------
    A tuple containing the container widget and the choice widget. The container widget
    is a QWidget that contains the label and the choice widget.
    """
    container = QWidget()
    container.setLayout(QHBoxLayout())
    container.layout().addWidget(QLabel(label))
    choice_list = create_widget(
        widget_type="ComboBox",
        name=name,
        value=value,
        options=options,
    )
    container.layout().addWidget(choice_list.native)
    return container, choice_list


class SelectFrom2DHistogram:
    def __init__(self, parent, ax, full_data):
        self.parent = parent
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.xys = full_data

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
        self.ind_mask = []

    def onselect(self, verts):
        path = Path(verts)

        self.ind_mask = path.contains_points(self.xys)
        self.ind = np.nonzero(self.ind_mask)[0]

        if self.parent.manual_clustering_method is not None:
            self.parent.manual_clustering_method(self.ind_mask)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


# Class below was based upon matplotlib lasso selection example:
# https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, parent, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect, button=1)
        self.ind = []
        self.ind_mask = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind_mask = path.contains_points(self.xys)
        self.ind = np.nonzero(self.ind_mask)[0]

        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.selected_coordinates = self.xys[self.ind].data

        if self.parent.manual_clustering_method is not None:
            self.parent.manual_clustering_method(self.ind_mask)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, manual_clustering_method=None):
        self.fig = Figure(figsize=(width, height), constrained_layout=True)
        self.manual_clustering_method = manual_clustering_method

        self.axes = self.fig.add_subplot(111)
        self.histogram = None

        self.match_napari_layout()
        self.xylim = None

        super().__init__(self.fig)
        self.mpl_connect("draw_event", self.on_draw)
        # polygons for 2d histogram
        self.polygons = []
        self.pts = self.axes.scatter([], [])
        self.selector = SelectFromCollection(self, self.axes, self.pts)
        self.rectangle_selector = RectangleSelector(
            self.axes,
            self.draw_rectangle,
            useblit=True,
            props=dict(edgecolor="white", fill=False),
            button=3,  # right button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.reset()

    def reset_zoom(self):
        if self.xylim:
            self.axes.set_xlim(self.xylim[0])
            self.axes.set_ylim(self.xylim[1])

    def on_draw(self, event):
        self.xylim = (self.axes.get_xlim(), self.axes.get_ylim())

    def draw_rectangle(self, eclick, erelease):
        """eclick and erelease are the press and release events"""
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        self.xys = self.pts.get_offsets()
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        self.rect_ind_mask = [
            min_x <= x <= max_x and min_y <= y <= max_y
            for x, y in zip(self.xys[:, 0], self.xys[:, 1])
        ]
        if self.manual_clustering_method is not None:
            self.manual_clustering_method(self.rect_ind_mask)

    def reset(self):
        self.axes.clear()
        self.is_pressed = None

    def hide_all_polygons(self):
        for p in self.polygons:
            p.remove()
        self.axes.figure.canvas.draw_idle()

    def make_2d_histogram(
        self,
        data_x: "numpy.typing.ArrayLike",
        data_y: "numpy.typing.ArrayLike",
        colors: "typing.List[str]",
        bin_number: int = 400,
        log_scale: bool = False,
    ):
        self.colors = colors
        norm = None
        if log_scale:
            norm = "log"
        h, xedges, yedges = np.histogram2d(data_x, data_y, bins=bin_number)
        self.axes.imshow(
            h.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="magma",
            aspect="auto",
            norm=norm,
        )
        self.axes.set_xlim(xedges[0], xedges[-1])
        self.axes.set_ylim(yedges[0], yedges[-1])
        # h, xedges, yedges, _ = self.axes.hist2d(data_x, data_y, bins=bin_number, cmap="magma", norm=norm, alpha=1)
        self.histogram = (h, xedges, yedges)

        full_data = pd.concat([data_x, data_y], axis=1)
        self.selector.disconnect()
        self.selector = SelectFrom2DHistogram(self, self.axes, full_data)
        self.axes.figure.canvas.draw_idle()

    def show_polygons(self):
        for poly_i, poly in enumerate(self.polygons):
            c = self.colors[int(poly_i + 1) % len(self.colors)]
            poly.set_facecolor(c)
            self.axes.add_patch(poly)
        self.axes.figure.canvas.draw_idle()

    def make_scatter_plot(
        self,
        data_x: "numpy.typing.ArrayLike",
        data_y: "numpy.typing.ArrayLike",
        colors: "typing.List[str]",
        sizes: "typing.List[float]",
        alpha: "typing.List[float]",
    ):
        self.pts = self.axes.scatter(
            data_x,
            data_y,
            c=colors,
            s=sizes,
            alpha=alpha,
        )
        self.selector.disconnect()
        self.selector = SelectFromCollection(
            self,
            self.axes,
            self.pts,
        )

    def match_napari_layout(self):
        """Change background and axes colors to match napari layout"""
        # changing color of axes background to napari main window color
        self.fig.patch.set_facecolor("#262930")
        # changing color of plot background to napari main window color
        self.axes.set_facecolor("#262930")

        # changing colors of all axes
        self.axes.spines["bottom"].set_color("white")
        self.axes.spines["top"].set_color("white")
        self.axes.spines["right"].set_color("white")
        self.axes.spines["left"].set_color("white")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")

        # changing colors of axes ticks
        self.axes.tick_params(axis="x", colors="white", labelcolor="white")
        self.axes.tick_params(axis="y", colors="white", labelcolor="white")

        # changing colors of axes labels
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")


# overriding NavigationToolbar method to change the background and axes colors of saved figure
class MyNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.canvas = canvas

    def _update_buttons_checked(self):
        super()._update_buttons_checked()
        # changes pan/zoom icons depending on state (checked or not)
        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Pan_checked.png"))
                )
            else:
                self._actions["pan"].setIcon(QIcon(os.path.join(ICON_ROOT, "Pan.png")))
        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Zoom_checked.png"))
                )
            else:
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Zoom.png"))
                )

    def save_figure(self):
        # setting the background of the saved figure to white
        self.canvas.fig.set_facecolor("#ffffff")
        self.canvas.fig.axes[0].set_facecolor("#ffffff")

        # setting axes colors of the saved figure to black
        self.canvas.axes.spines["bottom"].set_color("black")
        self.canvas.axes.spines["top"].set_color("black")
        self.canvas.axes.spines["right"].set_color("black")
        self.canvas.axes.spines["left"].set_color("black")

        # changing colors of axes ticks and labels for the saved figure
        self.canvas.axes.tick_params(axis="x", colors="black")
        self.canvas.axes.tick_params(axis="y", colors="black")

        self.canvas.axes.xaxis.label.set_color("black")
        self.canvas.axes.yaxis.label.set_color("black")

        super().save_figure()

        self.canvas.match_napari_layout()

        self.canvas.draw()
