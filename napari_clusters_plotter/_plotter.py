
import pyclesperanto_prototype as cle
import pandas as pd
import numpy as np
import warnings
# import hdbscan
import napari
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from napari_plugin_engine import napari_hook_implementation
from PyQt5 import QtWidgets
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QSpinBox, QHBoxLayout, QVBoxLayout, QComboBox, QGridLayout, \
    QFileDialog, QTableWidget, QTableWidgetItem
from qtpy.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.use('Qt5Agg')

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=7, height=4):
        self.fig = Figure(figsize=(width, height))

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
        # self.axes.title('UMAP projection')

        # changing colors of axis labels
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')

        super(MplCanvas, self).__init__(self.fig)

        # a rectangle defined via an anchor point xy and its width and height.
        self.rect = Rectangle((0, 0), 1, 1, edgecolor='white', fill=None)
        self.is_pressed = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.axes.add_patch(self.rect)

        # add an event when the user clicks somewhere in the plot
        self.mpl_connect('button_press_event', self._on_left_click)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self.on_release)

    # draws a dot where user clicks on the map
    # def _on_left_click(self, event):
    #     print("clicked at", event.xdata, event.ydata)
    #     self.axes.scatter(event.xdata, event.ydata)
    #     self.fig.canvas.draw()

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
    def on_release(self, event):
        self.is_pressed = False
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()
        coordinates = [self.x0, self.y0, self.x1, self.y1]

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

class PlotterWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # a figure instance to plot on
        self.figure = Figure()

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance as a parameter to __init__
        self.graphics_widget = MplCanvas(self.figure)

        # Navigation widget
        self.toolbar = MyNavigationToolbar(self.graphics_widget, self)

        # create a placeholder widget to hold the toolbar and graphics widget.
        graph_container = QWidget()
        graph_container.setMaximumHeight(500)
        graph_container.setLayout(QtWidgets.QVBoxLayout())
        graph_container.layout().addWidget(self.toolbar)
        graph_container.layout().addWidget(self.graphics_widget)

        # setup layout of the whole dialog. QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)

        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())

        label_plotting = QLabel("<b>Plotting</b>")
        label_container.layout().addWidget(label_plotting)

        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())

        # selection of labels layer
        label_label_list = QLabel("Labels layer")
        self.label_list = QComboBox()

        # update axes combo boxes once a label is selected
        self.label_list.currentIndexChanged.connect(self.update_axes_list)

        choose_img_container.layout().addWidget(label_label_list)
        choose_img_container.layout().addWidget(self.label_list)

        self.plot_x_axis = QComboBox()
        self.plot_y_axis = QComboBox()


        self.update_label_list()

        # selection if region properties should be calculated now or uploaded from file
        reg_props_container = QWidget()
        reg_props_container.setLayout(QHBoxLayout())
        label_axes = QLabel("Axes")
        reg_props_container.layout().addWidget(label_axes)
        reg_props_container.layout().addWidget(self.plot_x_axis)
        reg_props_container.layout().addWidget(self.plot_y_axis)

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        def run_clicked():

            self.run(
                self.get_selected_label().properties,
                self.plot_x_axis.currentText(),
                self.plot_y_axis.currentText()
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        # side note: if widget is not added to the layout but set visible by connecting an event,
        # it opens up as a pop-up

        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(reg_props_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

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

        print("Selected layer none?")
        if selected_layer is not None:
            print("Properties none?")
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                print("Add measurements")
                self.plot_x_axis.clear()
                self.plot_x_axis.addItems(list(properties.keys()))
                self.plot_y_axis.clear()
                self.plot_y_axis.addItems(list(properties.keys()))

    def _on_selection(self, event=None):

        num_labels_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

    # this function runs after the run button is clicked
    def run(self, properties, plot_x_axis_name, plot_y_axis_name):
        print("Plot running")

        if properties is None:
            warnings.warn("No labels image with properties was selected! Consider doing measurements first.")
            return

        self.data_x = properties[plot_x_axis_name]
        self.data_y = properties[plot_y_axis_name]

        self.graphics_widget.axes.scatter(self.data_x, self.data_y, color='#BABABA', s=10)
        self.graphics_widget.axes.set_aspect('equal', 'datalim')
        self.graphics_widget.draw()

        print('Plotting finished.')
