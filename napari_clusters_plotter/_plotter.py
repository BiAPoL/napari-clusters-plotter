import warnings
import napari
from PyQt5 import QtWidgets
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib.patches import Rectangle
from ._utilities import generate_parametric_cluster_image
from napari_tools_menu import  register_dock_widget


matplotlib.use('Qt5Agg')

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=7, height=4, manual_clustering_method=None):
        self.fig = Figure(figsize=(width, height))
        self.manual_clustering_method = manual_clustering_method

        # changing color of axis background to napari main window color
        self.fig.patch.set_facecolor('#262930')
        self.axes = self.fig.add_subplot(111)

        # changing color of plot background to napari main window color
        self.axes.set_facecolor('#262930')

        # changing colors of all axis
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['right'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')

        # changing colors of axis labels
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')

        super(MplCanvas, self).__init__(self.fig)

        # init rectangle defined via an anchor point xy and its width and height.
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.rect = Rectangle((0, 0), 1, 1, edgecolor='white', fill=None)

        self.reset()

        # add an event when the user clicks somewhere in the plot
        self.mpl_connect('button_press_event', self._on_left_click)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

    def reset(self):
        self.axes.clear()
        self.is_pressed = None
        self.axes.add_patch(self.rect)

    def _on_left_click(self, event):
        self.is_pressed = True
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()

    def _on_motion(self, event):
        if self.is_pressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.fig.canvas.draw()

    # draws a rectangle when user releases the mouse
    def _on_release(self, event):
        self.is_pressed = False
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()
        coordinates = [self.x0, self.y0, self.x1, self.y1]

        # if a function is configured to do the manual clustering, call it
        if self.manual_clustering_method is not None:
            self.manual_clustering_method(*coordinates)


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

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None
        self._available_labels = []

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_labels_layer = None

        def manual_clustering_method(x0, y0, x1, y1):
            print("Coords", x0, y0, x1, y1)
            if self.analysed_layer is None or x0 == x1 or y0 == y1:
                return # if nothing was plotted yet, leave

            min_x = min(x0, x1)
            max_x = max(x0, x1)
            min_y = min(y0, y1)
            max_y = max(y0, y1)

            clustering_ID = "MANUAL_CLUSTER_ID"

            # save manual clustering; for each point if it's inside the rectangle
            inside = [x >= min_x and x <= max_x and y >= min_y and y <= max_y for x, y in zip(self.data_x, self.data_y)]
            self.analysed_layer.properties[clustering_ID] = inside

            # redraw the whole plot
            self.run(self.analysed_layer.properties, self.plot_x_axis_name, self.plot_y_axis_name,
                     plot_cluster_name=clustering_ID)

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance
        self.graphics_widget = MplCanvas(self.figure, manual_clustering_method=manual_clustering_method)

        # Navigation widget
        self.toolbar = MyNavigationToolbar(self.graphics_widget, self)

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

        # selection of labels layer
        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())
        self.label_list = QComboBox()
        choose_img_container.layout().addWidget(QLabel("Labels layer"))
        choose_img_container.layout().addWidget(self.label_list)
        self.update_label_list()

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

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        def run_clicked():
            self.run(
                self.get_selected_label().properties,
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText(),
                self.plot_cluster_id.currentText()
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(axes_container)
        self.layout().addWidget(cluster_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # update axes combo boxes once a label is selected
        self.label_list.currentIndexChanged.connect(self.update_axes_list)

    def get_selected_label(self):
        index = self.label_list.currentIndex()
        if index >= 0:
            return self._available_labels[index]
        return None

    def update_label_list(self):
        selected_layer = self.get_selected_label()
        selected_index = -1

        self._available_labels = []
        self.label_list.clear()
        i = 0
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._available_labels.append(layer)
                if layer == selected_layer:
                    selected_index = i
                self.label_list.addItem(layer.name)
                i = i + 1
        self.label_list.setCurrentIndex(selected_index)

    def update_axes_list(self):
        selected_layer = self.get_selected_label()

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

    def _on_selection(self, event=None):

        num_labels_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

    # this function runs after the run button is clicked
    def run(self, properties, plot_x_axis_name, plot_y_axis_name, plot_cluster_name=None):
        if properties is None:
            warnings.warn("No labels image with properties was selected! Consider doing measurements first.")
            return
        if plot_x_axis_name is None or plot_y_axis_name is None:
            warnings.warn("No axis(-es) was/were selected!")
            return

        self.data_x = properties[plot_x_axis_name]
        self.data_y = properties[plot_y_axis_name]
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.plot_cluster_name = plot_cluster_name
        self.analysed_layer = self.get_selected_label()

        self.graphics_widget.reset()

        if plot_cluster_name is not None and plot_cluster_name != "label" and plot_cluster_name in list(properties.keys()):
            self.cluster_ids = properties[plot_cluster_name]

            color = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']
            self.graphics_widget.axes.scatter(
                self.data_x,
                self.data_y,
                c=[color[int(x)] for x in self.cluster_ids],
                cmap='Spectral',
                s=10
            )

            cluster_ids_in_space = generate_parametric_cluster_image(self.analysed_layer.data, self.cluster_ids)

            if self.visualized_labels_layer is None:
                self.visualized_labels_layer = self.viewer.add_labels(cluster_ids_in_space)
            else:
                self.visualized_labels_layer.data = cluster_ids_in_space
            if self.visualized_labels_layer not in self.viewer.layers:
                self.visualized_labels_layer = self.viewer.add_labels(self.visualized_labels_layer.data)

        else:
            self.graphics_widget.axes.scatter(self.data_x, self.data_y, color='#BABABA', s=10)
        self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
        self.graphics_widget.axes.set_ylabel(plot_y_axis_name)

        self.graphics_widget.draw()
