import pyclesperanto_prototype as cle
import pandas as pd
import numpy as np
import warnings
import napari
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QComboBox, QLineEdit
from ._utilities import show_table, widgets_inactive
from napari_tools_menu import  register_dock_widget

@register_dock_widget(menu="Measurement > Measure intensity, shape and neighbor counts (ncp)")
class MeasureWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Measurement</b>"))

        # selection of image layer
        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())
        self.image_list = QComboBox()
        self.update_image_list()
        choose_img_container.layout().addWidget(QLabel("Image layer"))
        choose_img_container.layout().addWidget(self.image_list)

        # selection of labels layer
        self.label_list = QComboBox()
        self.update_label_list()
        choose_img_container.layout().addWidget(QLabel("Labels layer"))
        choose_img_container.layout().addWidget(self.label_list)

        # selection if region properties should be measured now or uploaded from file
        reg_props_container = QWidget()
        reg_props_container.setLayout(QHBoxLayout())
        reg_props_container.layout().addWidget(QLabel("Region Properties"))
        self.reg_props_choice_list = QComboBox()
        self.reg_props_choice_list.addItems(['   ', 'Measure now (intensity)', 'Measure now (shape)',
                                             'Measure now (intensity + shape)',
                                             'Measure now (intensity + shape with neighborhood data)', 'Upload file'])
        reg_props_container.layout().addWidget(self.reg_props_choice_list)

        # region properties file upload
        self.regpropsfile_widget = QWidget()
        self.regpropsfile_widget.setLayout(QVBoxLayout())
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_FILE,
            filter='*.csv',
            value="  ")
        self.regpropsfile_widget.layout().addWidget(filename_edit.native)
        self.regpropsfile_widget.setVisible(False)

        # average distance of n closest points list
        self.closest_points_container = QWidget()
        self.closest_points_container.setLayout(QHBoxLayout())
        self.closest_points_container.layout().addWidget(QLabel("Average distance of n closest points list"))
        self.closest_points_list = QLineEdit()
        self.closest_points_container.layout().addWidget(self.closest_points_list)
        self.closest_points_list.setText("2, 3, 4")
        self.closest_points_container.setVisible(False)

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
                self.closest_points_list.text(),
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(reg_props_container)
        self.layout().addWidget(self.regpropsfile_widget)
        self.layout().addWidget(self.closest_points_container)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.reg_props_choice_list.currentIndexChanged.connect(self.change_reg_props_file)
        self.reg_props_choice_list.currentIndexChanged.connect(self.change_closest_points_list)

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

    def change_closest_points_list(self):
        widgets_inactive(self.closest_points_container,
                         active=self.reg_props_choice_list.currentText() == 'Measure now (with neighborhood data)')

    # this function runs after the run button is clicked
    def run(self, image, labels, regpropsfile, n_closest_points_str):
        print("Measurement running")

        labels_layer = self.get_selected_label()

        # depending on settings,...
        region_props_source = self.reg_props_choice_list.currentText()

        if region_props_source == 'Upload file':
            # load region properties from csv file
            reg_props = pd.read_csv(regpropsfile)
            try:
                edited_regprops = reg_props.drop(['Unnamed: 0'], axis = 1)
            except KeyError:
                edited_regprops = reg_props
            
            if 'labels' not in edited_regprops.keys().tolist():
                label_column = pd.DataFrame({'label':np.array(range(len(edited_regprops)))})
                reg_props_w_labels = pd.concat([label_column,edited_regprops], axis = 1)
                
            labels_layer.properties = reg_props_w_labels

        elif 'Measure now' in region_props_source:
            # or determine it now using clEsperanto
            reg_props = cle.statistics_of_labelled_pixels(image, labels)

            # and select columns, depending on if intensities and/or shape were selected
            columns = ['label', 'centroid_x', 'centroid_y', 'centroid_z']
            if 'intensity' in region_props_source:
                columns = columns + ['min_intensity', 'max_intensity', 'sum_intensity',
                                     'mean_intensity', 'standard_deviation_intensity']

            if 'shape' in region_props_source:
                columns = columns + ['area', 'mean_distance_to_centroid',
                                     'max_distance_to_centroid', 'mean_max_distance_to_centroid_ratio']

            if 'shape' in region_props_source or 'intensity' in region_props_source:
                reg_props = {column: value for column, value in reg_props.items() if column in columns}

                # saving measurement results into the properties of the analysed labels layer
                labels_layer.properties = reg_props

            if 'neighborhood' in region_props_source:

                n_closest_points_split = n_closest_points_str.split(",")
                n_closest_points_list = map(int, n_closest_points_split)
                print("regionprops_with_neighborhood_data function got " + str(columns))
                reg_props = regionprops_with_neighborhood_data(columns, labels, n_closest_points_list, reg_props)

                labels_layer.properties = reg_props

            print("Measured:", list(reg_props.keys()))

        else:
            warnings.warn("No measurements.")
            return

        show_table(self.viewer, labels_layer)


def regionprops_with_neighborhood_data(columns, label_image, n_closest_points_list, reg_props):

    # get lowest label index to adjust sizes of measurement arrays
    min_label = int(np.min(label_image[np.nonzero(label_image)]))

    columns = columns + ['min_intensity', 'max_intensity', 'sum_intensity',
               'mean_intensity', 'standard_deviation_intensity', 'area', 'mean_distance_to_centroid',
               'max_distance_to_centroid', 'mean_max_distance_to_centroid_ratio']

    region_props = {column: value for column, value in reg_props.items() if column in columns}

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

    # conversion and editing of the distance matrix, so that it does not break cle.average_distance
    viewdist_mat = cle.pull(distance_matrix)
    tempdist_mat = np.delete(viewdist_mat, range(min_label), axis=0)
    edited_distmat = np.delete(tempdist_mat, range(min_label), axis=1)

    # iterating over different neighbor numbers for average neighbor distance calculation
    for i in n_closest_points_list:
        distance_of_n_closest_points = \
            cle.pull(cle.average_distance_of_n_closest_points(cle.push(edited_distmat), n=i))[0]

        # addition to the regionprops dictionary
        region_props['avg distance of {val} closest points'.format(val=i)] = distance_of_n_closest_points

    # processing touching neighbor count for addition to regionprops (deletion of background & not used labels)
    touching_neighbor_c = cle.pull(touching_neighbor_count)
    touching_neighbor_count_formatted = np.delete(touching_neighbor_c, list(range(min_label)))

    # addition to the regionprops dictionary
    region_props['touching neighbor count'] = touching_neighbor_count_formatted
    print('Measurements Completed.')

    return region_props
