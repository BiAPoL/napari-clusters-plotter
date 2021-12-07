import warnings
import napari
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
from napari_tools_menu import  register_dock_widget
from qtpy.QtCore import QTimer
import pandas as pd

matplotlib.use('Qt5Agg')
# Class below was based upon matplotlib lasso selection example (https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html)
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
        # print("Got these points: ", self.selected_coordinates)
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

        self.pts = self.axes.scatter([],[])
        self.selector = SelectFromCollection(self, self.axes, self.pts)
        self.rectangle_selector = RectangleSelector(self.axes, self.draw_rectangle, 
                                      drawtype='box', useblit=True,
                                      rectprops = dict(edgecolor="white", fill=False),
                                      button=3,  # right button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=False)
        self.reset()
    
    def draw_rectangle(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        self.xys = self.pts.get_offsets()
        min_x  = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        self.rect_ind_mask = [x >= min_x and x <= max_x and y >= min_y and y <= max_y for x, y in zip(self.xys[:,0], self.xys[:,1])]
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

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None
        self._available_labels = []

        # a figure instance to plot on
        self.figure = Figure()

        self.analysed_layer = None
        self.visualized_labels_layer = None

        def manual_clustering_method(inside):
            if self.analysed_layer is None or len(inside)==0:
                return # if nothing was plotted yet, leave
            clustering_ID = "MANUAL_CLUSTER_ID"
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

        # adding spacing between fields for selecting two axes
        axes_container.layout().setSpacing(6)

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

    def clicked_label_in_view(self, event, event1):
        # We need to run this lagter as the labels_layer.selected_label isn't changed yet.
        QTimer.singleShot(200, self.after_clicked_label_in_view)

    def after_clicked_label_in_view(self):
        clustering_ID = "MANUAL_CLUSTER_ID"

        # save manual clustering; select only the label that's currently seleted on the layer
        inside = np.ones((self.analysed_layer.data.max()))
        inside[self.analysed_layer.selected_label - 1] = 0
        self.analysed_layer.properties[clustering_ID] = inside

        self.run(self.analysed_layer.properties, self.plot_x_axis_name, self.plot_y_axis_name, plot_cluster_name=clustering_ID)

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

            # get long colormap from function
            colors = get_nice_colormap()
            
            self.graphics_widget.pts = self.graphics_widget.axes.scatter(
                self.data_x,
                self.data_y,
                c=[colors[int(x)] for x in self.cluster_ids],
                cmap='Spectral',
                # here spot size is set differentially: larger (5) for all clustered datapoints (id >=0)
                # and smaller (2.5) for the noise points with id = -1
                s=[5 if id >= 0 else 2.5 for id in self.cluster_ids],

                # here alpha is set differentially: higher (0.7) for all clustered datapoints (id >= 0)
                # and lower (0.3) for the noise points with id = -1
                alpha = [0.7 if id >= 0 else 0.3 for id in self.cluster_ids]
            )
            self.graphics_widget.selector.disconnect()
            self.graphics_widget.selector = SelectFromCollection(self.graphics_widget,self.graphics_widget.axes, 
                                                                 self.graphics_widget.pts)

            # measuring time to check if speedup is worth it
            import time
            start = time.process_time()

            # get colormap as rgba array with background (see through) at beginning
            cmap = hex_colormap_to_list(colors)
            np_cluster_ids_p1 = np.array(self.cluster_ids) + 1

            # generate dictionary mapping each label to the color of the cluster
            cluster_id_dict = {i+1 : cmap[int(color)] for i,color in enumerate(np_cluster_ids_p1)}
            end_cl_dict_generation = time.process_time()

            keep_selection = list(self.viewer.layers.selection)
            if self.visualized_labels_layer is None:
                # instead of visualising cluster image, visualise label image with dictionary mapping
                self.visualized_labels_layer = self.viewer.add_labels(self.analysed_layer.data, color = cluster_id_dict)
            else:
                # instead of updating data, update colormap
                self.visualized_labels_layer.color = cluster_id_dict
            if self.visualized_labels_layer not in self.viewer.layers:
                self.visualized_labels_layer = self.viewer.add_labels(self.analysed_layer.data, color = cluster_id_dict)

            end = time.process_time()
            print('showing cluster layer took {}s'.format(end-start))
            print('generating cmap dictionary took {}s'.format(end_cl_dict_generation-start))

            self.viewer.layers.selection.clear()
            for s in keep_selection:
                self.viewer.layers.selection.add(s)


        else:
            self.graphics_widget.pts = self.graphics_widget.axes.scatter(self.data_x, self.data_y, color='#BABABA', s=5)
            self.graphics_widget.selector = SelectFromCollection(self.graphics_widget,self.graphics_widget.axes, 
                                                                 self.graphics_widget.pts)
            self.graphics_widget.draw() # Only redraws when cluster is not manually selected
                                        #   because manual selection already does that elsewhere
        self.graphics_widget.axes.set_xlabel(plot_x_axis_name)
        self.graphics_widget.axes.set_ylabel(plot_y_axis_name)


        # remove interaction from all label layers, just in case
        for layer in self.viewer.layers:
            if self.clicked_label_in_view in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self.clicked_label_in_view)

        self.analysed_layer.mouse_drag_callbacks.append(self.clicked_label_in_view)

def hex_colormap_to_list(hex_color_list):
    hex_color_list_w_background = ['#000000'] + hex_color_list
    rgba_palette_list = [list(int(h[i:i+2], 16)/255 for i in (1, 3, 5))+[1] for h in hex_color_list_w_background]
    rgba_palette_list[0][3] = 0

    return rgba_palette_list

def get_nice_colormap():
    colours_w_old_colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ccebc5', '#ffed6f', '#0054b6',
                '#6aa866', '#ffbfff', '#8d472a', '#417239', '#d48fd0', '#8b7e32', '#7989dc',
                '#f1d200', '#a1e9f6', '#924c28', '#dc797e', '#b86e85', '#79ea30', '#4723b9',
                '#3de658', '#de3ce7', '#86e851', '#9734d7', '#d0f23c', '#3c4ce7', '#93d229',
                '#8551e9', '#eeea3c', '#ca56ee', '#2af385', '#ea48cd', '#7af781', '#7026a8',
                '#51d967', '#ad3bc2', '#4ab735', '#3b1784', '#afc626', '#3d44bc', '#d5cc31',
                '#6065e6', '#8fca40', '#9e2399', '#27ca6f', '#e530a4', '#54f2ad', '#c236aa',
                '#a1e76b', '#a96fe6', '#64a725', '#d26de1', '#52b958', '#867af4', '#ecbe2b',
                '#4f83f7', '#bbd14f', '#2f65d0', '#ddf47c', '#27165e', '#92e986', '#8544ad',
                '#91a824', '#2e8bf3', '#ec6e1b', '#2b6abe', '#eb3e22', '#43e8cf', '#e52740',
                '#5ef3e7', '#ed2561', '#6ceac0', '#681570', '#8eec9c', '#8f2071', '#add465',
                '#3a4093', '#e3ce58', '#5a3281', '#82bf5d', '#e1418b', '#3d8e2a', '#e86ec2',
                '#66ca7d', '#ae1e63', '#4abb81', '#dc3b6c', '#409e59', '#b34b9d', '#87a943',
                '#958df3', '#e59027', '#667edb', '#ddad3c', '#545daf', '#e4e68b', '#22123e',
                '#b9e997', '#6c2c76', '#b0c163', '#866ecb', '#5f892d', '#d889e2', '#276222',
                '#ab98ed', '#79801a', '#8f5baa', '#ab972e', '#7899e9', '#dc5622', '#4a9de3',
                '#bd2e10', '#54d5d6', '#bc2f25', '#40bd9c', '#c72e45', '#9ae5b4', '#891954',
                '#d6ecb1', '#0e0d2c', '#e9c779', '#193163', '#f07641', '#4ab5dc', '#e35342',
                '#6dd3e7', '#92230d', '#a3e9e2', '#951a28', '#48a7b4', '#a8421a', '#88c4e9',
                '#c55a2b', '#2e5c9d', '#bb8524', '#737bc6', '#c2bc64', '#661952', '#92bc82',
                '#46123b', '#d6e5c8', '#190b1f', '#e5a860', '#1d1d3c', '#f27c58', '#06121f',
                '#ebcfa3', '#06121f', '#f3a27d', '#06121f', '#eb6065', '#297a53', '#af437c',
                '#365412', '#be9ee2', '#636b24', '#e9a1d5', '#1c2c0c', '#e3bce6', '#06121f',
                '#cf8042', '#06121f', '#bfdee0', '#751718', '#80c1ab', '#bb3f44', '#2b9083',
                '#781731', '#618d58', '#93457c', '#7f954c', '#4b2a5c', '#c3bd83', '#290d1b',
                '#ced0ec', '#6a2d0a', '#9db5ea', '#a35c1b', '#4781b1', '#9e4e22', '#33547a',
                '#876a1c', '#514e80', '#a59952', '#b86198', '#1d3621', '#eb7ba2', '#002a33',
                '#e38273', '#17212e', '#e8c4c5', '#281c2e', '#b3b18a', '#581430', '#659c84',
                '#a23a50', '#2d7681', '#a44634', '#608ea2', '#783121', '#94a9bc', '#4b1615',
                '#a4ae9f', '#7c3258', '#aa8242', '#7a6ea2', '#5f5621', '#c27dae', '#403911',
                '#a499c7', '#805124', '#717e9e', '#b8644f', '#143b44', '#ce6472', '#142a25',
                '#dd9ca6', '#21344a', '#d7a78c', '#3c3551', '#928853', '#ad486c', '#3a4d2d',
                '#8c5481', '#516b4d', '#994440', '#2e5667', '#af7e5c', '#432432', '#b49bb0',
                '#382718', '#b67576', '#294d46', '#935c54', '#52756e', '#6d363c', '#85856a',
                '#644466', '#635738', '#876d84', '#623c23', '#596776', '#864e5d', '#5f5848',
                '#9f7e80', '#5c4a56', '#735647', '#bcbcbc']

    return colours_w_old_colors