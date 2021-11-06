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
from ._utilities import show_table


def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


class MeasureWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # setup layout of the whole dialog. QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())

        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())

        label_measure = QLabel("<b>Measurement</b>")
        label_container.layout().addWidget(label_measure)

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

        def run_clicked():

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
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide choose file widget unless Upload file option is chosen
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
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._available_labels.append(layer)
                if layer == selected_layer:
                    selected_index = i
                self.label_list.addItem(layer.name)
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

        num_labels_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

        num_images_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)])
        if num_images_in_viewer != self.image_list.size():
            self.update_image_list()

    # toggle widgets visibility according to what is selected

    def change_reg_props_file(self):
        widgets_inactive(self.regpropsfile_widget,
                         active=self.reg_props_choice_list.currentText() == 'Upload file')

    # this function runs after the run button is clicked
    def run(self, image, labels, regpropsfile):
        print("Measurement running")

        labels_layer = self.get_selected_label()

        # depending on settings,...
        region_props_source = self.reg_props_choice_list.currentText()

        if region_props_source == 'Upload file':
            # load regions region properties from file
            reg_props = pd.read_csv(regpropsfile)
            labels_layer.properties = reg_props

        elif 'Measure now' in region_props_source:
            # or determine it now using clEsperanto
            reg_props = cle.statistics_of_labelled_pixels(image, labels)

            # and select columns, depending on if intensities and/or shape were selected
            columns = ['label']
            if 'intensity' in region_props_source:
                columns = columns + ['min_intensity', 'max_intensity', 'sum_intensity',
                                     'mean_intensity', 'standard_deviation_intensity']

            if 'shape' in region_props_source:
                columns = columns + ['area', 'mean_distance_to_centroid',
                                     'max_distance_to_centroid', 'mean_max_distance_to_centroid_ratio']

            if columns:
                reg_props = {column: value for column, value in reg_props.items() if column in columns}

                # add table widget to napari
                labels_layer.properties = reg_props

            if 'neighborhood' in region_props_source:
                reg_props = regionprops_with_neighborhood_data(labels, image, n_closest_points_list=[2, 3, 4])
                labels_layer.properties = reg_props

            print("Measured:", list(reg_props.keys()))
        else:
            warnings.warn("No measurements.")
            return

        show_table(self.viewer, labels_layer)


def regionprops_with_neighborhood_data(label_image, original_image, n_closest_points_list):
    from skimage.measure import regionprops_table

    # get lowest label index to adjust sizes of measurement arrays
    min_label = int(np.min(label_image[np.nonzero(label_image)]))

    # defining function for getting standard deviation as an extra property
    # arguments must be in the specified order, matching regionprops
    def image_stdev(region, intensities):
        # note the ddof arg to get the sample var if you so desire!
        return np.std(intensities[region], ddof=1)

    # get region properties from labels
    regionprops = regionprops_table(label_image.astype(dtype='uint16'), intensity_image=original_image,
                                    properties=('area', 'centroid', 'feret_diameter_max',
                                                'major_axis_length', 'minor_axis_length', 'solidity',
                                                'mean_intensity',
                                                'max_intensity', 'min_intensity'),
                                    extra_properties=[image_stdev])
    print('Scikit Regionprops Done')

    # determine neighbors of cells
    touch_matrix = cle.generate_touch_matrix(label_image)

    # ignore touching the background
    cle.set_column(touch_matrix, 0, 0)
    cle.set_row(touch_matrix, 0, 0)

    # determine distances of all cells to all cells
    pointlist = cle.centroids_of_labels(label_image)

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
    touching_neighbor_count_formatted = np.delete(touching_neighbor_c, list(range(min_label)))

    # addition to the regionprops dictionary
    regionprops['touching neighbor count'] = touching_neighbor_count_formatted
    print('Regionprops Completed')

    return regionprops


