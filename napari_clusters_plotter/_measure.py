import warnings
from enum import Enum

import dask.array as da
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from magicgui.types import FileDialogMode
from magicgui.widgets import FileEdit, create_widget
from napari.layers import Image, Labels
from napari.qt.threading import create_worker
from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

from ._utilities import set_features, show_table, widgets_inactive


@register_dock_widget(
    menu="Measurement > Measure intensity, shape and neighbor counts (ncp)"
)
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
        self.reg_props_choice_list = create_widget(
            widget_type="ComboBox",
            name="Region_properties",
            value=self.Choices.EMPTY.value,
            options=dict(choices=[e.value for e in self.Choices]),
        )

        reg_props_container.layout().addWidget(self.reg_props_choice_list.native)

        # region properties file upload
        self.reg_props_file_widget = QWidget()
        self.reg_props_file_widget.setLayout(QVBoxLayout())
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_FILE, filter="*.csv", value="  "
        )
        self.reg_props_file_widget.layout().addWidget(filename_edit.native)
        self.reg_props_file_widget.setVisible(False)

        # average distance of n closest points list
        self.closest_points_container = QWidget()
        self.closest_points_container.setLayout(QHBoxLayout())
        self.closest_points_container.layout().addWidget(
            QLabel("Average distance of n closest points list")
        )
        self.closest_points_list = QLineEdit()
        self.closest_points_container.layout().addWidget(self.closest_points_list)
        self.closest_points_list.setText("2, 3, 4")
        self.closest_points_container.setVisible(False)

        # checkbox whether image is a timelapse
        self.timelapse_container = QWidget()
        self.timelapse_container.setLayout(QHBoxLayout())
        self.timelapse = create_widget(
            widget_type="CheckBox",
            name="Timelapse",
            value=False,
        )

        self.timelapse_container.layout().addWidget(self.timelapse.native)

        # Run button
        run_button_container = QWidget()
        run_button_container.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_button_container.layout().addWidget(run_button)

        # Progress bar
        progress_bar_container = QWidget()
        progress_bar_container.setLayout(QHBoxLayout())
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()
        progress_bar_container.layout().addWidget(self.progress_bar)

        # adding all widgets to the layout
        self.layout().addWidget(title_container)
        self.layout().addWidget(image_layer_selection_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(reg_props_container)
        self.layout().addWidget(self.reg_props_file_widget)
        self.layout().addWidget(self.closest_points_container)
        self.layout().addWidget(self.timelapse_container)
        self.layout().addWidget(run_button_container)
        self.layout().setSpacing(0)

        def run_clicked():
            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if self.image_select.value is None:
                warnings.warn("No image was selected!")
                return

            if (
                not self.timelapse.value
                and len(self.image_select.value.data.shape) == 4
            ):
                warnings.warn("Please check that timelapse checkbox is checked!")
                return

            self.run(
                self.image_select.value,
                self.labels_select.value,
                self.reg_props_choice_list.value,
                str(filename_edit.value.absolute())
                .replace("\\", "/")
                .replace("//", "/"),
                self.closest_points_list.text(),
                self.timelapse.value,
            )

        run_button.clicked.connect(run_clicked)
        run_button.clicked.connect(self.progress_bar_status)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

        self.layout().addWidget(progress_bar_container)

        # hide widgets unless appropriate options are chosen
        self.reg_props_choice_list.changed.connect(self.change_reg_props_file)
        self.reg_props_choice_list.changed.connect(self.change_closest_points_list)

    def progress_bar_status(self):
        self.progress_bar.show()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.image_select.reset_choices(event)
        self.labels_select.reset_choices(event)

    # toggle widgets visibility according to what is selected
    def change_reg_props_file(self):
        widgets_inactive(
            self.reg_props_file_widget,
            active=self.reg_props_choice_list.value == self.Choices.FILE.value,
        )

    def change_closest_points_list(self):
        widgets_inactive(
            self.closest_points_container,
            active=self.reg_props_choice_list.value == self.Choices.NEIGHBORHOOD.value,
        )

    # this function runs after the run button is clicked
    def run(
        self,
        image_layer,
        labels_layer,
        region_props_source,
        reg_props_file,
        n_closest_points_str,
        timelapse,
    ):
        print("Measurement running")
        print("Region properties source: " + str(region_props_source))

        def result_of_get_regprops(returned):
            # saving measurement results into the properties or features of the analysed labels layer
            # df = pd.DataFrame(returned)
            # df_without_bg = df.drop([0])
            set_features(labels_layer, returned)
            print("Measured:", list(returned.keys()))
            show_table(self.viewer, labels_layer)
            self.progress_bar.hide()

        # depending on settings,...
        if region_props_source == self.Choices.FILE.value:
            # load region properties from csv file
            reg_props = pd.read_csv(reg_props_file)
            try:
                edited_reg_props = reg_props.drop(["Unnamed: 0"], axis=1)
            except KeyError:
                edited_reg_props = reg_props

            if "label" not in edited_reg_props.keys().tolist():
                label_column = pd.DataFrame(
                    {"label": np.array(range(1, (len(edited_reg_props) + 1)))}
                )
                reg_props_w_labels = pd.concat([label_column, edited_reg_props], axis=1)
                set_features(labels_layer, reg_props_w_labels)
            else:
                set_features(labels_layer, edited_reg_props)
            show_table(self.viewer, labels_layer)
            self.progress_bar.hide()

        elif "Measure now" in region_props_source:
            if "shape" in region_props_source or "intensity" in region_props_source:
                self.worker = create_worker(
                    new_regprops,
                    intensity_image=image_layer.data,
                    label_image=labels_layer.data,
                    timelapse=timelapse,
                    region_props_source=region_props_source,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_get_regprops)
                self.worker.start()

            if "neighborhood" in region_props_source:
                n_closest_points_split = n_closest_points_str.split(",")
                n_closest_points_list = map(int, n_closest_points_split)

                self.worker = create_worker(
                    new_regprops,
                    intensity_image=image_layer.data,
                    label_image=labels_layer.data,
                    region_props_source=region_props_source,
                    timelapse=timelapse,
                    n_closest_points_list=n_closest_points_list,
                    _progress=True,
                )
                self.worker.returned.connect(result_of_get_regprops)
                self.worker.start()

        else:
            warnings.warn("No measurements.")
            return

def new_regprops(
    intensity_image,
    label_image,
    region_props_source,
    timelapse,
    n_closest_points_list=[2, 3, 4],
    ):

    print("Shape of the intensity image: " + str(intensity_image.shape))
    print("Shape of the labels image: " + str(label_image.shape))

    # and select columns, depending on if intensities, neighborhood 
    # and/or shape were selected
    columns = ["label", "centroid_x", "centroid_y", "centroid_z"]
    intensity_columns = [
            "min_intensity",
            "max_intensity",
            "sum_intensity",
            "mean_intensity",
            "standard_deviation_intensity",
        ]
    shape_columns = [
            "area",
            "mean_distance_to_centroid",
            "max_distance_to_centroid",
            "mean_max_distance_to_centroid_ratio",
        ]
    if "intensity" in region_props_source:
        columns += intensity_columns
    if "shape" in region_props_source:
        columns += shape_columns
    if "neighborhood" in region_props_source:
        columns += shape_columns
        columns += intensity_columns

    if timelapse:
        reg_props_all = []
        for t, timepoint in enumerate(tqdm(range(intensity_image.shape[0]))):
            all_reg_props_single_t = pd.DataFrame(
                cle.statistics_of_labelled_pixels(
                    intensity_image[timepoint], label_image[timepoint]
                )
            )

            if "neighborhood" in region_props_source:
                reg_props_single_t = region_props_with_neighborhood_data(
                    label_image[timepoint], 
                    n_closest_points_list, 
                    all_reg_props_single_t[columns]
                )
            else:
                reg_props_single_t = all_reg_props_single_t[columns]

            timepoint_column = pd.DataFrame(
                {"timepoint":np.full(len(reg_props_single_t),t)}
                )
            reg_props_with_tp_column = pd.concat(
                [reg_props_single_t,timepoint_column], axis = 1
                )
            reg_props_all.append(reg_props_with_tp_column)

        reg_props = pd.concat(reg_props_all)
        print("Reg props measured for each timepoint.")
        return reg_props

    reg_props = pd.DataFrame(cle.statistics_of_labelled_pixels(intensity_image, label_image))
    print("Reg props not measured for a timelapse.")
    if "neighborhood" in region_props_source:
        return region_props_with_neighborhood_data(
                    label_image, n_closest_points_list, reg_props[columns]
                )
        
    return reg_props[columns]


def get_regprops_from_regprops_source(
    intensity_image,
    label_image,
    region_props_source,
    timelapse,
    n_closest_points_list=[2, 3, 4],
):
    """
    Calculate Region properties based on the region properties source string

    Parameters
    ----------
    timelapse : bool
        true if original image is a timelapse
    intensity_image : numpy array
        original image from which the labels were generated
    label_image : numpy array
        segmented image with background = 0 and labels >= 1
    region_props_source: str
        must include either shape, intensity, both or neighborhood
    n_closest_points_list: list
        number of closest neighbors for which neighborhood properties will be calculated
    """
    print("Shape of the intensity image: " + str(intensity_image.shape))
    print("Shape of the labels image: " + str(label_image.shape))

    # and select columns, depending on if intensities and/or shape were selected
    columns = ["label", "centroid_x", "centroid_y", "centroid_z"]

    if "intensity" in region_props_source:
        columns = columns + [
            "min_intensity",
            "max_intensity",
            "sum_intensity",
            "mean_intensity",
            "standard_deviation_intensity",
        ]

    if "shape" in region_props_source:
        columns = columns + [
            "area",
            "mean_distance_to_centroid",
            "max_distance_to_centroid",
            "mean_max_distance_to_centroid_ratio",
        ]

    # Determine Region properties using clEsperanto
    # if the input is a timelapse iterate over each timepoint and
    # determine regionprops
    if timelapse:
        reg_props_all = []
        for t, timepoint in enumerate(tqdm(range(intensity_image.shape[0]))):
            reg_props_single_t = pd.DataFrame(
                cle.statistics_of_labelled_pixels(
                    intensity_image[timepoint], label_image[timepoint]
                )
            )
            if "neighborhood" in region_props_source:
                reg_props_single_t = region_props_with_neighborhood_data(
                    label_image, n_closest_points_list, reg_props_single_t
                )
            else:
                reg_props_single_t = reg_props_single_t[columns]
            reg_props_single_t["timepoint"] = t
            reg_props_all.append(reg_props_single_t)

        reg_props = pd.concat(reg_props_all)
        print("Reg props measured for each timepoint.")
        return reg_props
    
    else:
        reg_props = cle.statistics_of_labelled_pixels(intensity_image, label_image)
        print("Reg props measured not for a timelapse.")

    if "shape" in region_props_source or "intensity" in region_props_source:
        return {
            column: value for column, value in reg_props.items() if column in columns
        }

    if "neighborhood" in region_props_source:
        return region_props_with_neighborhood_data(
            label_image, n_closest_points_list, reg_props
        )


def region_props_with_neighborhood_data(
    label_image, n_closest_points_list, reg_props
):
    """
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
    """
    neighborhood_properties = {}
    if isinstance(label_image, da.core.Array):
        label_image = np.asarray(label_image)
    
    # get the lowest label index to adjust sizes of measurement arrays
    min_label = int(np.min(label_image[np.nonzero(label_image)]))

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
        distance_of_n_closest_points = cle.pull(
            cle.average_distance_of_n_closest_points(cle.push(edited_dist_mat), n=i)
        )[0]

        # addition to the regionprops dictionary
        neighborhood_properties[
            f"avg distance of {i} closest points"
        ] = distance_of_n_closest_points

    # processing touching neighbor count for addition to regionprops (deletion of background & not used labels)
    touching_neighbor_c = cle.pull(touching_neighbor_count)
    touching_neighbor_count_formatted = np.delete(
        touching_neighbor_c, list(range(min_label))
    )

    # addition to the regionprops dictionary
    neighborhood_properties["touching neighbor count"] = touching_neighbor_count_formatted
    return pd.concat(
        [reg_props,pd.DataFrame(neighborhood_properties)],axis = 1
        )
