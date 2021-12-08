import warnings
import napari
from napari.layers import Labels
from PyQt5 import QtWidgets
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
from ._utilities import generate_parametric_cluster_image
from napari_tools_menu import register_dock_widget
from qtpy.QtCore import QTimer
<<<<<<< Updated upstream
from magicgui.widgets import create_widget
=======
from qtpy.QtGui import QIcon
>>>>>>> Stashed changes

matplotlib.use('Qt5Agg')


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
            raise ValueError('Collection must have a facecolor')
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
        self.fig.patch.set_facecolor('#262930')
        self.axes = self.fig.add_subplot(111)

        # changing color of plot background to napari main window color
        self.axes.set_facecolor('#262930')

        # changing colors of all axes
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['right'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')

        # changing colors of axes labels
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')

        super(MplCanvas, self).__init__(self.fig)

        self.pts = self.axes.scatter([], [])
        self.selector = SelectFromCollection(self, self.axes, self.pts)
        self.rectangle_selector = RectangleSelector(self.axes, self.draw_rectangle,
                                                    drawtype='box', useblit=True,
                                                    rectprops=dict(edgecolor="white", fill=False),
                                                    button=3,  # right button
                                                    minspanx=5, minspany=5,
                                                    spancoords='pixels',
                                                    interactive=False)
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
        self.rect_ind_mask = [min_x <= x <= max_x and min_y <= y <= max_y for x, y in
                              zip(self.xys[:, 0], self.xys[:, 1])]
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
    
    def save_figure(self):
        self.canvas.fig.set_facecolor("#00000000")
        self.canvas.fig.axes[0].set_facecolor("#00000000")
        self.canvas.axes.tick_params(color='black')

        self.canvas.axes.spines['bottom'].set_color('black')
        self.canvas.axes.spines['top'].set_color('black')
        self.canvas.axes.spines['right'].set_color('black')
        self.canvas.axes.spines['left'].set_color('black')

        # changing colors of axis labels
        self.canvas.axes.tick_params(axis='x', colors='black')
        self.canvas.axes.tick_params(axis='y', colors='black')

        super().save_figure()

        self.canvas.axes.tick_params(color='white')

        self.canvas.axes.spines['bottom'].set_color('white')
        self.canvas.axes.spines['top'].set_color('white')
        self.canvas.axes.spines['right'].set_color('white')
        self.canvas.axes.spines['left'].set_color('white')

        # changing colors of axis labels
        self.canvas.axes.tick_params(axis='x', colors='white')
        self.canvas.axes.tick_params(axis='y', colors='white')

        self.canvas.draw()


@register_dock_widget(menu="Measurement > Plot measurements (ncp)")
class PlotterWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_labels_layer = None

        # noinspection PyPep8Naming
        def manual_clustering_method(inside):
            if self.analysed_layer is None or len(inside) == 0:
                return  # if nothing was plotted yet, leave
            clustering_ID = "MANUAL_CLUSTER_ID"
            self.analysed_layer.properties[clustering_ID] = inside

            # redraw the whole plot
            self.run(self.analysed_layer.properties, self.plot_x_axis_name, self.plot_y_axis_name,
                     plot_cluster_name=clustering_ID)

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance
        self.graphics_widget = MplCanvas(self.figure, manual_clustering_method=manual_clustering_method)

        # Navigation widget
        self.toolbar = MyNavigationToolbar(self.graphics_widget, self)
        
        # Modify toolbar icons and some tooltips
        for action in self.toolbar.actions():
            text = action.text()
            if text == 'Pan':
                action.setToolTip('Left button pans, Right button zooms\nClick to activate\nClick again to deactivate')
            if text == 'Zoom':
                action.setToolTip("Zoom to rectangle\nClick to activate\nClick again to deactivate")
            if len(text)>0:
                icon_path = "images//my_toolbar_icons//" + text + ".png"
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

        # selection if region properties should be calculated now or uploaded from file
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
        update_button = QPushButton("Update Axes Lists")
        update_container.layout().addWidget(update_button)

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_widget.layout().addWidget(run_button)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("Please select labels layer!")
                return
            if self.labels_select.value.properties is None:
                warnings.warn("No labels image with properties was selected! Consider doing measurements first.")
                return
            if self.plot_x_axis.currentText() == '' or self.plot_y_axis.currentText() == '':
                warnings.warn('No axis(-es) was/were selected! If you cannot see anything in axes selection boxes, '
                              'but you have performed measurements/dimensionality reduction before, try clicking '
                              'Update Axes List')
                return

            self.run(
                self.labels_select.value.properties,
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText(),
                self.plot_cluster_id.currentText()
            )

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(axes_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(cluster_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # adding spacing between fields for selecting two axes
        axes_container.layout().setSpacing(6)

        # update axes combo boxes once a new label layer is selected
        self.labels_select.changed.connect(self.update_axes_list)

        # update axes combo boxes once a update button is clicked
        update_button.clicked.connect(self.update_axes_list)

        # select what happens when the run button is clicked
        run_button.clicked.connect(run_clicked)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)

    def clicked_label_in_view(self, event, event1):
        # We need to run this lagter as the labels_layer.selected_label isn't changed yet.
        QTimer.singleShot(200, self.after_clicked_label_in_view)

    def after_clicked_label_in_view(self):
        clustering_ID = "MANUAL_CLUSTER_ID"

        # save manual clustering; select only the label that's currently selected on the layer
        inside = np.ones((self.analysed_layer.data.max()))
        inside[self.analysed_layer.selected_label - 1] = 0
        self.analysed_layer.properties[clustering_ID] = inside

        self.run(self.analysed_layer.properties, self.plot_x_axis_name, self.plot_y_axis_name,
                 plot_cluster_name=clustering_ID)

    def update_axes_list(self):
        selected_layer = self.labels_select.value

        former_x_axis = self.plot_x_axis.currentIndex()
        former_y_axis = self.plot_y_axis.currentIndex()
        former_cluster_id = self.plot_cluster_id.currentIndex()

        if selected_layer is not None:
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                self.plot_x_axis.clear()
                self.plot_x_axis.addItems(list(properties.keys()))
                self.plot_y_axis.clear()
                self.plot_y_axis.addItems(list(properties.keys()))
                self.plot_cluster_id.clear()
                self.plot_cluster_id.addItems([l for l in list(properties.keys()) if "CLUSTER" in l])
        self.plot_x_axis.setCurrentIndex(former_x_axis)
        self.plot_y_axis.setCurrentIndex(former_y_axis)
        self.plot_cluster_id.setCurrentIndex(former_cluster_id)

    # this function runs after the run button is clicked
    def run(self, properties, plot_x_axis_name, plot_y_axis_name, plot_cluster_name=None):

        self.data_x = properties[plot_x_axis_name]
        self.data_y = properties[plot_y_axis_name]
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.plot_cluster_name = plot_cluster_name
        self.analysed_layer = self.labels_select.value

        self.graphics_widget.reset()

        if plot_cluster_name is not None and plot_cluster_name != "label" and plot_cluster_name in list(
                properties.keys()):
            self.cluster_ids = properties[plot_cluster_name]

            color = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']
            self.graphics_widget.pts = self.graphics_widget.axes.scatter(
                self.data_x,
                self.data_y,
                c=[color[int(x)] for x in self.cluster_ids],
                cmap='Spectral',
                s=10
            )
            self.graphics_widget.selector.disconnect()
            self.graphics_widget.selector = SelectFromCollection(self.graphics_widget, self.graphics_widget.axes,
                                                                 self.graphics_widget.pts)

            cluster_ids_in_space = generate_parametric_cluster_image(self.analysed_layer.data, self.cluster_ids)

            keep_selection = list(self.viewer.layers.selection)
            if self.visualized_labels_layer is None:
                self.visualized_labels_layer = self.viewer.add_labels(cluster_ids_in_space)
            else:
                self.visualized_labels_layer.data = cluster_ids_in_space
            if self.visualized_labels_layer not in self.viewer.layers:
                self.visualized_labels_layer = self.viewer.add_labels(self.visualized_labels_layer.data)

            self.viewer.layers.selection.clear()
            for s in keep_selection:
                self.viewer.layers.selection.add(s)

        else:
            self.graphics_widget.pts = self.graphics_widget.axes.scatter(self.data_x, self.data_y, color='#BABABA',
                                                                         s=10)
            self.graphics_widget.selector = SelectFromCollection(self.graphics_widget, self.graphics_widget.axes,
                                                                 self.graphics_widget.pts)
            self.graphics_widget.draw()  # Only redraws when cluster is not manually selected
            # because manual selection already does that elsewhere
        self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
        self.graphics_widget.axes.set_ylabel(plot_y_axis_name)

        # remove interaction from all label layers, just in case
        for layer in self.viewer.layers:
            if self.clicked_label_in_view in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self.clicked_label_in_view)

        self.analysed_layer.mouse_drag_callbacks.append(self.clicked_label_in_view)
