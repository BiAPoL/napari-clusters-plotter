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
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QSpinBox, QHBoxLayout, QVBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.use('Qt5Agg')

'''
To do list:
1) add HDBSCAN
2) add PCA
3) sparse PCA
4) highlight regions of the image (approximated cells) corresponding to user-selected areas in the plot or vice versa??
'''


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return Widget, dict(name='Clustering and Plotting')


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
        self.rect = Rectangle((0, 0), 1, 1, edgecolor = 'white', fill = None)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.axes.add_patch(self.rect)

        # add an event when the user clicks somewhere in the plot
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        # self.mpl_connect("button_press_event", self._on_left_click)

    # draws a dot where user clicks on the map
    def _on_left_click(self, event):
        print("clicked at", event.xdata, event.ydata)
        self.axes.scatter(event.xdata, event.ydata)
        self.fig.canvas.draw()

    # initial coordinates x0, y0 (anchor point) for the rectangle
    def _on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    # draws a rectangle when user releases the mouse
    def _on_release(self, event):
        print('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()


# overwriting NavigationToolbar class to change the background and axes colors of saved figure
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


class Widget(QWidget):

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

        label_clustr = QLabel("<b>Clustering</b>")
        label_container.layout().addWidget(label_clustr)

        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())

        # selection of image layer
        label_image_list = QLabel("Image layer")
        self.image_list = QComboBox()
        self.update_image_list()

        choose_img_container.layout().addWidget(label_image_list)
        choose_img_container.layout().addWidget(self.image_list)

        # selection of labels layer
        label_label_list = QLabel("Labels layer")
        self.label_list = QComboBox()
        self.update_label_list()

        choose_img_container.layout().addWidget(label_label_list)
        choose_img_container.layout().addWidget(self.label_list)

        # selection if region properties should be calculated now or uploaded from file
        reg_props_container = QWidget()
        reg_props_container.setLayout(QHBoxLayout())
        label_reg_props = QLabel("Region Properties")
        reg_props_container.layout().addWidget(label_reg_props)

        self.reg_props_choice_list = QComboBox()
        self.reg_props_choice_list.addItems(['   ', 'Measure now (with neighborhood data)', 'Measure now (intensity)',
                                             'Measure now (shape)', 'Measure now (intensity + shape)', 'Upload file'])

        reg_props_container.layout().addWidget(self.reg_props_choice_list)

        # selection of the clustering methods
        self.clust_method_container = QWidget()
        self.clust_method_container.setLayout(QHBoxLayout())
        label_clust_method = QLabel("Clustering Method")
        self.clust_method_container.layout().addWidget(label_clust_method)

        self.clust_method_choice_list = QComboBox()
        self.clust_method_choice_list.addItems(['   ', 'KMeans', 'HDBSCAN'])
        self.clust_method_container.layout().addWidget(self.clust_method_choice_list)

        # clustering options for KMeans
        # selection of number of clusters
        self.kmeans_settings_container = QWidget()
        self.kmeans_settings_container.setLayout(QHBoxLayout())
        label_kmeans_nr_clusters = QLabel("Number of Clusters")
        self.kmeans_settings_container.layout().addWidget(label_kmeans_nr_clusters)

        self.kmeans_nr_clusters = QSpinBox()
        self.kmeans_nr_clusters.setMinimumWidth(40)
        self.kmeans_nr_clusters.setMinimum(2)
        self.kmeans_nr_clusters.setValue(2)
        self.kmeans_settings_container.layout().addWidget(self.kmeans_nr_clusters)
        self.kmeans_settings_container.setVisible(False)

        # selection of number of iterations
        self.kmeans_settings_container2 = QWidget()
        self.kmeans_settings_container2.setLayout(QHBoxLayout())
        label_iter_nr = QLabel("Number of Iterations")
        self.kmeans_settings_container2.layout().addWidget(label_iter_nr)

        self.kmeans_nr_iter = QSpinBox()
        self.kmeans_nr_iter.setMinimumWidth(40)
        self.kmeans_nr_iter.setMinimum(1)
        self.kmeans_nr_iter.setMaximum(10000)
        self.kmeans_nr_iter.setValue(3000)
        self.kmeans_settings_container2.layout().addWidget(self.kmeans_nr_iter)
        self.kmeans_settings_container2.setVisible(False)

        # Clustering options for HDBSCAN

        # Region properties file upload
        self.regpropsfile_widget = QWidget()
        self.regpropsfile_widget.setLayout(QVBoxLayout())
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_FILE,
            filter='*.csv',
            value="  ")
        self.regpropsfile_widget.layout().addWidget(filename_edit.native)
        self.regpropsfile_widget.setVisible(False)

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        def run_clicked(*arg):

            if self.get_selected_label() is None:
                warnings.warn("No labels image was selected!")
                return

            if self.get_selected_image() is None:
                warnings.warn("No image was selected!")
                return

            self.run(
                self.get_selected_image().data,
                self.get_selected_label().data,
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                self.clust_method_choice_list.currentText(),
                self.kmeans_nr_clusters.value(),
                self.kmeans_nr_iter.value(),
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        # side note: if widget is not added to the layout but set visible by connecting an event,
        # it opens up as a pop-up

        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(reg_props_container)
        self.layout().addWidget(self.regpropsfile_widget)
        self.layout().addWidget(self.clust_method_container)
        self.layout().addWidget(self.kmeans_settings_container)
        self.layout().addWidget(self.kmeans_settings_container2)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widget for the selection of parameters for KMeans unless Kmeans clustering method is chosen
        self.clust_method_choice_list.currentIndexChanged.connect(self.change_kmeans_clustering)

        # hide choose file widget unless Upload file is chosen
        self.reg_props_choice_list.currentIndexChanged.connect(self.change_reg_props_file)

    # following 5 functions for image layer or labels layer selection are from Robert Haase
    # napari-accelerated-pixel-and-object-classification (APOC)

    def get_selected_image(self):
        index = self.image_list.currentIndex()
        if index >= 0:
            return self._available_images[index]
        return None

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
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Labels):
                self._available_labels.append(l)
                if l == selected_layer:
                    selected_index = i
                self.label_list.addItem(l.name)
                i = i + 1
        self.label_list.setCurrentIndex(selected_index)

    def update_image_list(self):
        selected_layer = self.get_selected_image()
        selected_index = -1

        self._available_images = []
        self.image_list.clear()
        i = 0
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._available_images.append(layer)
                if layer == selected_layer:
                    selected_index = i
                self.image_list.addItem(layer.name)
                i = i + 1
        self.image_list.setCurrentIndex(selected_index)

    def _on_selection(self, event=None):

        num_labels_in_viewer = len([l for l in self.viewer.layers if isinstance(l, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

        num_images_in_viewer = len([l for l in self.viewer.layers if isinstance(l, napari.layers.Image)])
        if num_images_in_viewer != self.image_list.size():
            self.update_image_list()

    # toggle widgets visibility according to what is selected
    def widgets_inactive(self, *widgets, active):
        for widget in widgets:
            widget.setVisible(active)

    def change_kmeans_clustering(self):
        self.widgets_inactive(self.kmeans_settings_container, self.kmeans_settings_container2,
                              active=self.clust_method_choice_list.currentText() == 'KMeans')

    def change_reg_props_file(self):
        self.widgets_inactive(self.regpropsfile_widget,
                              active=self.reg_props_choice_list.currentText() == 'Upload file')

    # this function runs after the run button is clicked
    def run(self, image, labels, regpropsfile, cluster_method, nr_clusters, iterations):
        print("running")

        if image is None:
            warnings.warn("No image was selected!")
            return

        if labels is None:
            warnings.warn("No labels image was selected!")
            return

        # depending on settings,...
        region_props_source = self.reg_props_choice_list.currentText()
        if region_props_source == 'Upload file':
            # load regions region properties from file
            reg_props = pd.read_csv(regpropsfile)
        elif 'Measure now' in region_props_source:
            # or determine it now using clEsperanto

            # and select columns, depending on if intensities and/or shape were selected
            columns = []
            if 'intensity' in region_props_source:
                reg_props = pd.DataFrame(cle.statistics_of_labelled_pixels(image, labels))
                columns = columns + ['min_intensity', 'max_intensity', 'sum_intensity',
                                     'mean_intensity', 'standard_deviation_intensity']
            if 'shape' in region_props_source:
                reg_props = pd.DataFrame(cle.statistics_of_labelled_pixels(image, labels))
                columns = columns + ['area', 'mean_distance_to_centroid',
                                     'max_distance_to_centroid', 'mean_max_distance_to_centroid_ratio']
            if columns:
                reg_props = reg_props[columns]

            if 'neighborhood' in region_props_source:
                reg_props = pd.DataFrame(
                    regionprops_with_neighborhood_data(labels, image, n_closest_points_list=[2, 3, 4]))

            print(list(reg_props.keys()))
        else:
            warnings.warn("No measurements.")
            return

        # check which GPU is used
        print('GPU used: {val}'.format(val=cle.get_device().name))

        if cluster_method == 'KMeans':
            kmeans_predictions = kmeansclustering(reg_props, nr_clusters, iterations)
            print('KMeans predictions done.')
            kmeans_cluster_labels = generate_parametric_cluster_image(labels, kmeans_predictions)
            self.viewer.add_labels(kmeans_cluster_labels)

            embedding = umap(reg_props)

            self.graphics_widget.axes.scatter(embedding[:, 0], embedding[:, 1], color='#BABABA', s=10)
            self.graphics_widget.axes.set_aspect('equal', 'datalim')
            color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']
            self.graphics_widget.axes.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=[color[int(x)] for x in kmeans_predictions],
                cmap='Spectral',
                s=10
            )
            self.graphics_widget.draw()

            print('Clustering finished.')


def kmeansclustering(measurements, cluster_number, iterations):
    from sklearn.cluster import KMeans
    print('KMeans predictions started...')

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    y_pred = km.fit_predict(measurements)

    # saving prediction as a list for generating clustering image
    return y_pred


# function for generating image labelled by clusters given the label image and the cluster prediction list
def generate_parametric_cluster_image(labelimage, predictionlist):
    print('Generation of parametric cluster image started...')
    # reforming the prediction list this is done to account for cluster labels that start at 0
    # conveniently hdbscan labelling starts at -1 for noise, removing these from the labels
    predictionlist_new = np.array(predictionlist) + 1

    # this takes care of the background label that needs to be 0 as well as any other
    # labels that might have been accidentally deleted

    for i in range(int(np.min(labelimage[np.nonzero(labelimage)]))):
        predictionlist_new = np.insert(predictionlist_new, i, 0)

    # pushing of variables into GPU
    cle_list = cle.push(predictionlist_new)
    cle_labels = cle.push(labelimage)

    # generation of cluster label image
    parametric_image = cle.pull(cle.replace_intensities(cle_labels, cle_list))
    print('Generation of parametric cluster image finished...')
    return np.array(parametric_image, dtype="int64")


'''
def HDBSCAN_predictionlist(dataframe, n_min_samples=10, n_min_cluster=50, n_dimension_umap=2):

    # dataframe = pd.DataFrame(regionpropsdict)

    if umap_used:
        # using UMAP to generate a dimension reduced non linear version of regionprops
        clusterable_embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=n_dimension_umap,
            random_state=42,
        ).fit_transform(dataframe)
        hdbscan_labels = hdbscan.HDBSCAN(min_samples=n_min_samples, min_cluster_size=n_min_cluster).fit_predict(
            clusterable_embedding)

    else:
        hdbscan_labels = hdbscan.HDBSCAN(min_samples=n_min_samples, min_cluster_size=n_min_cluster).fit_predict(
            dataframe)

    return hdbscan_labels
'''


def umap(reg_props):
    from sklearn.preprocessing import StandardScaler
    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=133)
    scaled_regionprops = StandardScaler().fit_transform(reg_props)

    embedding = reducer.fit_transform(scaled_regionprops)

    return embedding


def regionprops_with_neighborhood_data(labelimage, originalimage, n_closest_points_list):
    from skimage.measure import regionprops_table

    # get lowest label index to adjust sizes of measurement arrays
    min_label = int(np.min(labelimage[np.nonzero(labelimage)]))

    # defining function for getting standarddev as extra property
    # arguments must be in the specified order, matching regionprops
    def image_stdev(region, intensities):
        # note the ddof arg to get the sample var if you so desire!
        return np.std(intensities[region], ddof=1)

    # get region properties from labels
    regionprops = regionprops_table(labelimage.astype(dtype='uint16'), intensity_image=originalimage,
                                    properties=('area', 'centroid', 'feret_diameter_max',
                                                'major_axis_length', 'minor_axis_length', 'solidity',
                                                'mean_intensity',
                                                'max_intensity', 'min_intensity'),
                                    extra_properties=[image_stdev])
    print('Scikit Regionprops Done')

    # determine neighbors of cells
    touch_matrix = cle.generate_touch_matrix(labelimage)

    # ignore touching the background
    cle.set_column(touch_matrix, 0, 0)
    cle.set_row(touch_matrix, 0, 0)

    # determine distances of all cells to all cells
    pointlist = cle.centroids_of_labels(labelimage)

    # generate a distance matrix
    distance_matrix = cle.generate_distance_matrix(pointlist, pointlist)

    # detect touching neighbor count
    touching_neighbor_count = cle.count_touching_neighbors(touch_matrix)
    cle.set_column(touching_neighbor_count, 0, 0)

    # conversion and editing of the distance matrix, so that it doesn't break cle.average_distance.....
    viewdist_mat = cle.pull(distance_matrix)
    tempdist_mat = np.delete(viewdist_mat, range(min_label), axis=0)
    edited_distmat = np.delete(tempdist_mat, range(min_label), axis=1)

    # iterating over different neighbor numbers for avg neighbor dist calculation
    for i in n_closest_points_list:
        distance_of_n_closest_points = \
            cle.pull(cle.average_distance_of_n_closest_points(cle.push(edited_distmat), n=i))[0]

        # addition to the regionprops dictionary
        regionprops['avg distance of {val} closest points'.format(val=i)] = distance_of_n_closest_points

    # processing touching neighbor count for addition to regionprops (deletion of background & not used labels)
    touching_neighbor_c = cle.pull(touching_neighbor_count)
    touching_neighborcount_formated = np.delete(touching_neighbor_c, list(range(min_label)))

    # addition to the regionprops dictionary
    regionprops['touching neighbor count'] = touching_neighborcount_formated
    print('Regionprops Completed')

    return regionprops
