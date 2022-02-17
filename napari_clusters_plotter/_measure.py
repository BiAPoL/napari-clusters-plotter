from enum import Enum
import pyclesperanto_prototype as cle
import pandas as pd
import numpy as np
import warnings
from napari.layers import Labels, Image
from magicgui.widgets import FileEdit, create_widget
from magicgui.types import FileDialogMode
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit
from ._utilities import show_table, widgets_inactive, set_features
from napari_tools_menu import register_dock_widget


@register_dock_widget(menu="Measurement > Measure intensity, shape and neighbor counts (ncp)")
class MeasureWidget(QWidget):
    class Choices(Enum):
        EMPTY = " "
        NEIGHBORHOOD = "Measure now (with neighborhood data)"
        INTENSITY = "Measure now (intensity)"
        SHAPE = "Measure now (shape)"
        BOTH = "Measure now (intensity + shape)"
        FILE = "Upload file"

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        title_container = QWidget()
        title_container.setLayout(QVBoxLayout())
        title_container.layout().addWidget(QLabel("<b>Measurement</b>"))

        self.image_select = create_widget(annotation=Image, label="image_layer")
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        # widget for the selection of image layer
        image_layer_selection_container = QWidget()
        image_layer_selection_container.setLayout(QHBoxLayout())
        image_layer_selection_container.layout().addWidget(QLabel("Image layer"))
        image_layer_selection_container.layout().addWidget(self.image_select.native)

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # selection if region properties should be measured now or uploaded from file
        reg_props_container = QWidget()
        reg_props_container.setLayout(QHBoxLayout())
        reg_props_container.layout().addWidget(QLabel("Region Properties"))
        self.reg_props_choice_list = create_widget(widget_type="ComboBox",
                                                   name="Region_properties",
                                                   value=self.Choices.EMPTY.value,
                                                   options=dict(choices=[e.value for e in self.Choices]))

        reg_props_container.layout().addWidget(self.reg_props_choice_list.native)

        # region properties file upload
        self.reg_props_file_widget = QWidget()
        self.reg_props_file_widget.setLayout(QVBoxLayout())
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_FILE,
            filter='*.csv',
            value="  ")
        self.reg_props_file_widget.layout().addWidget(filename_edit.native)
        self.reg_props_file_widget.setVisible(False)

        # average distance of n closest points list
        self.closest_points_container = QWidget()
        self.closest_points_container.setLayout(QHBoxLayout())
        self.closest_points_container.layout().addWidget(QLabel("Average distance of n closest points list"))
        self.closest_points_list = QLineEdit()
        self.closest_points_container.layout().addWidget(self.closest_points_list)
        self.closest_points_list.setText("2, 3, 4")
        self.closest_points_container.setVisible(False)

        # Run button
        run_button_container = QWidget()
        run_button_container.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        # adding all widgets to the layout
        self.layout().addWidget(title_container)
        self.layout().addWidget(image_layer_selection_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(reg_props_container)
        self.layout().addWidget(self.reg_props_file_widget)
        self.layout().addWidget(self.closest_points_container)
        self.layout().addWidget(run_button_container)
        self.layout().setSpacing(0)

        def run_clicked():
            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if self.image_select.value is None:
                warnings.warn("No image was selected!")
                return

            self.run(
                self.image_select.value,
                self.labels_select.value,
                self.reg_props_choice_list.value,
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                self.closest_points_list.text(),
            )

        button.clicked.connect(run_clicked)
        run_button_container.layout().addWidget(button)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        # hide widgets unless appropriate options are chosen
        self.reg_props_choice_list.changed.connect(self.change_reg_props_file)
        self.reg_props_choice_list.changed.connect(self.change_closest_points_list)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.image_select.reset_choices(event)
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_reg_props_file(self):
        widgets_inactive(self.reg_props_file_widget,
                         active=self.reg_props_choice_list.value == self.Choices.FILE.value)

    def change_closest_points_list(self):
        widgets_inactive(self.closest_points_container,
                         active=self.reg_props_choice_list.value == self.Choices.NEIGHBORHOOD.value)

    # this function runs after the run button is clicked
    def run(self, image_layer, labels_layer, region_props_source, reg_props_file, n_closest_points_str):
        print("Measurement running")
        print("Region properties source: " + str(region_props_source))

        # depending on settings,...

        if region_props_source == self.Choices.FILE.value:
            # load region properties from csv file
            reg_props = pd.read_csv(reg_props_file)
            try:
                edited_reg_props = reg_props.drop(['Unnamed: 0'], axis=1)
            except KeyError:
                edited_reg_props = reg_props

            if 'label' not in edited_reg_props.keys().tolist():
                label_column = pd.DataFrame({'label': np.array(range(1, (len(edited_reg_props) + 1)))})
                reg_props_w_labels = pd.concat([label_column, edited_reg_props], axis=1)
                set_features(labels_layer, reg_props_w_labels)
            else:
                set_features(labels_layer, edited_reg_props)

        elif 'Measure now' in region_props_source:
            if 'shape' in region_props_source or 'intensity' in region_props_source:
                reg_props = get_regprops_from_regprops_source(image_layer.data, 
                                                              labels_layer.data,
                                                              region_props_source)

                # saving measurement results into the properties or features of the analysed labels layer
                set_features(labels_layer, reg_props) 
                print("Measured:", list(reg_props.keys()))

            if 'neighborhood' in region_props_source:
                n_closest_points_split = n_closest_points_str.split(",")
                n_closest_points_list = map(int, n_closest_points_split)
                reg_props = get_regprops_from_regprops_source(image_layer.data, 
                                                              labels_layer.data,
                                                              region_props_source,
                                                              n_closest_points_list)

                set_features(labels_layer, reg_props)
                print("Measured:", list(reg_props.keys()))

        else:
            warnings.warn("No measurements.")
            return

        show_table(self.viewer, labels_layer)


def get_regprops_from_regprops_source(intensity_image, label_image, 
                                      region_props_source, 
                                      n_closest_points_list= [2,3,4]):
    '''
    Calculate Regionproperties based on the region properties source string

    Parameters
    ----------
    intensity_image : numpy array
        original image from which the labels were generated
    label_image : numpy array
        segmented image with background = 0 and labels >= 1
    region_props_source: str
        must include either shape, intensity, both or neighborhood
    n_closest_points_list: list
        number of closest neighbors for which neighborhood properties will be calculated
    '''
    # and select columns, depending on if intensities and/or shape were selected
    columns = ['label', 'centroid_x', 'centroid_y', 'centroid_z']

    if 'intensity' in region_props_source:
        columns = columns + ['min_intensity', 'max_intensity', 'sum_intensity',
                             'mean_intensity', 'standard_deviation_intensity']

    if 'shape' in region_props_source:
        columns = columns + ['area', 'mean_distance_to_centroid',
                             'max_distance_to_centroid', 
                             'mean_max_distance_to_centroid_ratio']

    # Determine Regionproperties using clEsperanto
    reg_props = cle.statistics_of_labelled_pixels(intensity_image, label_image)

    if 'shape' in region_props_source or 'intensity' in region_props_source: 
        return {column: value for column, value in reg_props.items() if column in columns}

    if 'neighborhood' in region_props_source:
        return region_props_with_neighborhood_data(columns, 
                                                   label_image, 
                                                   n_closest_points_list,
                                                   reg_props)

        

def region_props_with_neighborhood_data(columns, label_image, n_closest_points_list, reg_props):
    '''
    Calculate neighborhood regionproperties and combine with other regionproperties

    Parameters
    ----------
    columns: list
        list of names of regionproperties
    label_image : numpy array
        segmented image with background = 0 and labels >= 1
    reg_props: dict
        region properties to be combined with
    n_closest_points_list: list
        number of closest neighbors for which neighborhood properties will be calculated
    '''

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
    view_dist_mat = cle.pull(distance_matrix)
    temp_dist_mat = np.delete(view_dist_mat, range(min_label), axis=0)
    edited_dist_mat = np.delete(temp_dist_mat, range(min_label), axis=1)

    # iterating over different neighbor numbers for average neighbor distance calculation
    for i in n_closest_points_list:
        distance_of_n_closest_points = \
            cle.pull(cle.average_distance_of_n_closest_points(cle.push(edited_dist_mat), n=i))[0]

        # addition to the regionprops dictionary
        region_props['avg distance of {val} closest points'.format(val=i)] = distance_of_n_closest_points

    # processing touching neighbor count for addition to regionprops (deletion of background & not used labels)
    touching_neighbor_c = cle.pull(touching_neighbor_count)
    touching_neighbor_count_formatted = np.delete(touching_neighbor_c, list(range(min_label)))

    # addition to the regionprops dictionary
    region_props['touching neighbor count'] = touching_neighbor_count_formatted
    print('Measurements Completed.')

    return region_props
