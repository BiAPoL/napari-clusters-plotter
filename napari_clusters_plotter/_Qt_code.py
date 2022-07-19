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
from magicgui.widgets import create_widget
from napari.layers import Labels, Image

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from pathlib import Path as PathL

import numpy as np
import os


from qtpy.QtGui import QIcon

ICON_ROOT = PathL(__file__).parent / "icons"

def measurements_container_and_list():
    properties_container = QWidget()
    properties_container.setLayout(QVBoxLayout())
    properties_container.layout().addWidget(QLabel("Measurements"))
    properties_list = QListWidget()
    properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    properties_list.setGeometry(QRect(10, 10, 101, 291))
    properties_container.layout().addWidget(properties_list)

    return properties_container, properties_list

def labels_container_and_selection():
    labels_layer_selection_container = QWidget()
    labels_layer_selection_container.setLayout(QHBoxLayout())
    labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
    labels_select = create_widget(annotation=Labels, label="labels_layer")
    labels_layer_selection_container.layout().addWidget(labels_select.native)

    return labels_layer_selection_container, labels_select

def image_container_and_selection():
    image_select = create_widget(annotation=Image, label="image_layer")
    image_layer_selection_container = QWidget()
    image_layer_selection_container.setLayout(QHBoxLayout())
    image_layer_selection_container.layout().addWidget(QLabel("Image layer"))
    image_layer_selection_container.layout().addWidget(image_select.native)

    return image_layer_selection_container, image_select

def title(name: str):
    title_container = QWidget()
    title_container.setLayout(QVBoxLayout())
    title_container.layout().addWidget(QLabel(name))

    return title_container

def n_clusters_containter_and_selection(name:str,value:int,label:str="Number of Clusters"):
    cluster_n_container = QWidget()
    cluster_n_container.setLayout(QHBoxLayout())
    cluster_n_container.layout().addWidget(
        QLabel(label)
    )
    cluster_n_selection = create_widget(
        widget_type="SpinBox",
        name=name,
        value=value,
        options={"min": 2, "step": 1},
    )

    cluster_n_container.layout().addWidget(
        cluster_n_selection.native
    )
    cluster_n_container.setVisible(False)

    return cluster_n_container,cluster_n_selection


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
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.ind_mask = path.contains_points(self.xys)
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
        self.fig = Figure(figsize=(width, height))
        self.manual_clustering_method = manual_clustering_method

        # changing color of axes background to napari main window color
        self.fig.patch.set_facecolor("#262930")
        self.axes = self.fig.add_subplot(111)

        # changing color of plot background to napari main window color
        self.axes.set_facecolor("#262930")

        # changing colors of all axes
        self.axes.spines["bottom"].set_color("white")
        self.axes.spines["top"].set_color("white")
        self.axes.spines["right"].set_color("white")
        self.axes.spines["left"].set_color("white")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")

        # changing colors of axes labels
        self.axes.tick_params(axis="x", colors="white")
        self.axes.tick_params(axis="y", colors="white")

        super().__init__(self.fig)

        self.pts = self.axes.scatter([], [])
        self.selector = SelectFromCollection(self, self.axes, self.pts)
        self.rectangle_selector = RectangleSelector(
            self.axes,
            self.draw_rectangle,
            drawtype="box",
            useblit=True,
            rectprops=dict(edgecolor="white", fill=False),
            button=3,  # right button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.reset()

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
        self.canvas.fig.set_facecolor("#00000000")
        self.canvas.fig.axes[0].set_facecolor("#00000000")
        self.canvas.axes.tick_params(color="black")

        self.canvas.axes.spines["bottom"].set_color("black")
        self.canvas.axes.spines["top"].set_color("black")
        self.canvas.axes.spines["right"].set_color("black")
        self.canvas.axes.spines["left"].set_color("black")

        # changing colors of axis labels
        self.canvas.axes.tick_params(axis="x", colors="black")
        self.canvas.axes.tick_params(axis="y", colors="black")

        super().save_figure()

        self.canvas.axes.tick_params(color="white")

        self.canvas.axes.spines["bottom"].set_color("white")
        self.canvas.axes.spines["top"].set_color("white")
        self.canvas.axes.spines["right"].set_color("white")
        self.canvas.axes.spines["left"].set_color("white")

        # changing colors of axis labels
        self.canvas.axes.tick_params(axis="x", colors="white")
        self.canvas.axes.tick_params(axis="y", colors="white")

        self.canvas.draw()