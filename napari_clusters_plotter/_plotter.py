import os
import warnings
from pathlib import Path as PathL

import numpy as np
import pandas as pd
from magicgui.widgets import create_widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from napari.layers import Labels
from napari_tools_menu import register_dock_widget
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._plotter_utilities import clustered_plot_parameters, unclustered_plot_parameters
from ._utilities import (
    add_column_to_layer_tabular_data,
    dask_cluster_image_timelapse,
    generate_cluster_image,
    get_layer_tabular_data,
    get_nice_colormap,
)

# can be changed to frame or whatever we decide to use
POINTER = "frame"
ICON_ROOT = PathL(__file__).parent / "icons"


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


@register_dock_widget(menu="Measurement > Plot measurements (ncp)")
class PlotterWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.cluster_ids = None
        self.viewer = napari_viewer

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_labels_layer = None

        # noinspection PyPep8Naming
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

            # redraw the whole plot
            self.run(
                features,
                self.plot_x_axis_name,
                self.plot_y_axis_name,
                plot_cluster_name=clustering_ID,
            )

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
        graph_container.setMaximumHeight(500)
        graph_container.setLayout(QtWidgets.QVBoxLayout())
        graph_container.layout().addWidget(self.toolbar)
        graph_container.layout().addWidget(self.graphics_widget)

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(graph_container)

        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Plotting</b>"))

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

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

        # Update measurements button
        update_container = QWidget()
        update_container.setLayout(QHBoxLayout())
        update_button = QPushButton("Update Axes/Clustering Selection Boxes")
        update_container.layout().addWidget(update_button)

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_widget.layout().addWidget(run_button)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(axes_container)
        self.layout().addWidget(cluster_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
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
        self.labels_select.changed.connect(self.update_axes_list)

        # update axes combo boxes automatically if features of
        # layer are changed
        self.last_connected = None
        self.labels_select.changed.connect(self.activate_property_autoupdate)

        # update axes combo boxes once update button is clicked
        update_button.clicked.connect(self.update_axes_list)

        # select what happens when the run button is clicked
        run_button.clicked.connect(run_clicked)

        self.viewer.dims.events.current_step.connect(frame_changed)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    def activate_property_autoupdate(self):
        if self.last_connected is not None:
            self.last_connected.events.properties.disconnect(self.update_axes_list)
        self.labels_select.value.events.properties.connect(self.update_axes_list)
        self.last_connected = self.labels_select.value

    def update_axes_list(self):
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

    # this function runs after the run button is clicked
    def run(
        self,
        features,
        plot_x_axis_name,
        plot_y_axis_name,
        plot_cluster_name=None,
        redraw_cluster_image=True,
    ):
        if not self.isVisible():
            # don't redraw in case the plot is invisible anyway
            return

        self.data_x = features[plot_x_axis_name]
        self.data_y = features[plot_y_axis_name]
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.plot_cluster_name = plot_cluster_name
        self.analysed_layer = self.labels_select.value

        self.graphics_widget.reset()
        number_of_points = len(features)
        tracking_data = (
            len(self.analysed_layer.data.shape) == 4 and "frame" not in features.keys()
        )

        if (
            plot_cluster_name is not None
            and plot_cluster_name != "label"
            and plot_cluster_name in list(features.keys())
        ):
            # fill all prediction nan values with -1 -> turns them
            # into noise points
            self.cluster_ids = features[plot_cluster_name].fillna(-1)

            # get long colormap from function
            colors = get_nice_colormap()
            if len(self.analysed_layer.data.shape) == 4 and not tracking_data:
                frame_id = features[POINTER].tolist()
                current_frame = self.frame
            elif len(self.analysed_layer.data.shape) <= 3 or tracking_data:
                frame_id = None
                current_frame = None
            else:
                warnings.warn("Image dimensions too high for processing!")

            a, sizes, colors_plot = clustered_plot_parameters(
                cluster_id=self.cluster_ids,
                frame_id=frame_id,
                current_frame=current_frame,
                n_datapoints=number_of_points,
                color_hex_list=colors,
            )

            self.graphics_widget.pts = self.graphics_widget.axes.scatter(
                self.data_x,
                self.data_y,
                c=colors_plot,
                s=sizes,
                alpha=a,
            )
            self.graphics_widget.selector.disconnect()
            self.graphics_widget.selector = SelectFromCollection(
                self.graphics_widget,
                self.graphics_widget.axes,
                self.graphics_widget.pts,
            )

            # get colormap as rgba array
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
                for prediction in self.cluster_ids
            }
            # take care of background label
            cmap_dict[0] = [0, 0, 0, 0]

            keep_selection = list(self.viewer.layers.selection)

            # Generating the cluster image
            if redraw_cluster_image:
                # depending on the dimensionality of the data
                # generate the cluster image -> TODO change so possible
                # with 2D timelapse data
                if len(self.analysed_layer.data.shape) == 4:
                    if not tracking_data:
                        max_timepoint = features[POINTER].max() + 1

                        prediction_lists_per_timepoint = [
                            features.loc[features[POINTER] == i][
                                plot_cluster_name
                            ].tolist()
                            for i in range(int(max_timepoint))
                        ]
                    else:
                        prediction_lists_per_timepoint = [
                            features[plot_cluster_name].tolist()
                            for i in range(self.analysed_layer.data.shape[0])
                        ]

                    cluster_image = dask_cluster_image_timelapse(
                        self.analysed_layer.data, prediction_lists_per_timepoint
                    )

                elif len(self.analysed_layer.data.shape) <= 3:
                    cluster_image = generate_cluster_image(
                        self.analysed_layer.data, self.cluster_ids
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

            a, sizes, colors_plot = unclustered_plot_parameters(
                frame_id=frame_id,
                current_frame=current_frame,
                n_datapoints=number_of_points,
            )

            # Potting
            self.graphics_widget.pts = self.graphics_widget.axes.scatter(
                self.data_x,
                self.data_y,
                color=colors_plot,
                s=sizes,
                alpha=a,
            )
            self.graphics_widget.selector = SelectFromCollection(
                self.graphics_widget,
                self.graphics_widget.axes,
                self.graphics_widget.pts,
            )
            self.graphics_widget.draw()  # Only redraws when cluster is not manually selected
            # because manual selection already does that elsewhere
        self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
        self.graphics_widget.axes.set_ylabel(plot_y_axis_name)
