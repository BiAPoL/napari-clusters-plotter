from enum import Enum, auto

import napari
import numpy as np
from napari.utils.colormaps import ALL_COLORMAPS
from biaplotter import CanvasWidget  # TODO make local import with "."
from qtpy import uic
from qtpy.QtCore import Qt
from pathlib import Path
from qtpy.QtWidgets import (
    QComboBox,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class PlottingType(Enum):
    HISTOGRAM = auto()
    SCATTER = auto()


class PlotterWidget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()

        self.control_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "plotter_inputs.ui",
            self.control_widget,
        )

        self.viewer = napari_viewer

        # create a scroll area
        self.scrollArea = QScrollArea()
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumWidth(450)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.contents = QWidget()
        self.scrollArea.setWidget(self.contents)

        self.layout = QVBoxLayout(self.contents)
        self.layout.setAlignment(Qt.AlignTop)

        self.plotting_widget = CanvasWidget(napari_viewer, self)

        # Add plot and options as widgets
        self.layout.addWidget(self.plotting_widget)
        self.layout.addWidget(self.control_widget)

        # Setting of Widget options
        self.hue: QComboBox = self.control_widget.hue_box

        self.control_widget.plot_type_box.addItems(
            [PlottingType.SCATTER.name, PlottingType.HISTOGRAM.name]
        )

        self.control_widget.cmap_box.addItems(list(ALL_COLORMAPS.keys()))
        self.control_widget.cmap_box.setCurrentIndex(
            np.argwhere(np.array(list(ALL_COLORMAPS.keys())) == "magma")[0][0]
        )

        # Setting Visibility Defaults
        self.control_widget.manual_bins_container.setVisible(False)
        self.control_widget.bins_settings_container.setVisible(False)
        self.control_widget.log_scale_container.setVisible(False)

        # Adding Connections
        self.control_widget.plot_type_box.currentIndexChanged.connect(
            plotting_type_changed
        )
        self.control_widget.set_bins_button.clicked.connect(bin_number_set)
        self.control_widget.auto_bins_checkbox.stateChanged.connect(bin_auto)
        self.control_widget.log_scale_checkbutton.stateChanged.connect(replot)
        self.control_widget.non_selected_checkbutton.stateChanged.connect(
            checkbox_status_changed
        )
        self.control_widget.cmap_box.currentIndexChanged.connect(replot)
        self.control_widget.x_axis_box.currentIndexChanged.connect(
            replot
        )  # TODO Decide if this is a good idea
        self.control_widget.y_axis_box.currentIndexChanged.connect(
            replot
        )  # TODO Decide if this is a good idea

        # initialising all variables
        self.log_scale = self.control_widget.log_scale_checkbutton.isChecked()
        self.automatic_bins = self.control_widget.auto_bins_checkbox.isChecked()
        # self.bin_number = self.control_widget.n_bins_box.value()
        self.hide_non_selected = (
            self.control_widget.non_selected_checkbutton.isChecked()
    def _replot(self):
        
        # if no x or y axis is selected, return
        if self.x_axis == '' or self.y_axis == '':
            return
        
        data_to_plot = self._get_data()
        self.plotting_widget.active_artist.data = data_to_plot
        # redraw the whole plot
        try:
            # plotting function needs to be here
            pass

        except AttributeError:
            # In this case, replotting is not yet possible
            pass

    def _checkbox_status_changed(self):
        self._replot()

    def _plotting_type_changed(self):  # TODO NEED TO ADD WHICH VARIABLE STORES THE TYPE
        if (
            self.control_widget.plot_type_box.currentText()
            == PlottingType.HISTOGRAM.name
        ):
            self.control_widget.bins_settings_container.setVisible(True)
            self.control_widget.log_scale_container.setVisible(True)
        elif (
            self.control_widget.plot_type_box.currentText()
            == PlottingType.SCATTER.name
        ):
            self.control_widget.bins_settings_container.setVisible(False)
            self.control_widget.log_scale_container.setVisible(False)

        self._replot()

    def _bin_number_set(self):
        self._replot()

    def _bin_auto(self):
        self.control_widget.manual_bins_container.setVisible(
            not self.control_widget.auto_bins_checkbox.isChecked()
        )
        if self.control_widget.auto_bins_checkbox.isChecked():
            self._replot()

    # Connecting the widgets to actual object variables:
    # using getters and setters for flexibility
    @property
    def log_scale(self):
        return self.control_widget.log_scale_checkbutton.isChecked()

    @log_scale.setter
    def log_scale(self, val: bool):
        self.control_widget.log_scale_checkbutton.setChecked(val)

    @property
    def automatic_bins(self):
        return self.control_widget.auto_bins_checkbox.isChecked()

    @automatic_bins.setter
    def automatic_bins(self, val: bool):
        self.control_widget.auto_bins_checkbox.setChecked(val)

    @property
    def bin_number(self):
        return self.control_widget.n_bins_box.value()

    @property
    def hide_non_selected(self):
        return self.control_widget.non_selected_checkbutton.isChecked()

    @hide_non_selected.setter
    def hide_non_selected(self, val: bool):
        self.control_widget.non_selected_checkbutton.setChecked(val)

    @property
    def colormap_plot(self):
        return self.control_widget.cmap_box.currentText()

    @property
    def plotting_type(self):
        return self.control_widget.plot_type_box.currentText()

    @plotting_type.setter
    def plotting_type(self, type):
        if type in PlottingType.__members__.keys():
            self.control_widget.plot_type_box.setCurrentText(type)

    @property
    def x_axis(self):
        return self.control_widget.x_axis_box.currentText()

    @x_axis.setter
    def x_axis(self, column: str):
        self.control_widget.x_axis_box.setCurrentText(
            column
        )
        self._replot()

    @property
    def y_axis(self):
        return self.control_widget.y_axis_box.currentText()

    @y_axis.setter
    def y_axis(self, column: str):
        self.control_widget.y_axis_box.setCurrentText(
            column
        )
        self._replot()

viewer = napari.Viewer()
widget = PlotterWidget(viewer)
viewer.window.add_dock_widget(widget)

# print("hi")
