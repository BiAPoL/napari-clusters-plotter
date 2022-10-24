import warnings
import numpy as np
import pandas as pd

from magicgui.types import FileDialogMode
from magicgui.widgets import FileEdit, create_widget
from napari.layers import Labels

from napari_tools_menu import register_dock_widget
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._utilities import set_features, show_table, widgets_inactive


@register_dock_widget(
    menu="Measurement > Import Measurements from File (ncp)"
)
class ImportWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.worker = None
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        title_container = QWidget()
        title_container.setLayout(QVBoxLayout())
        title_container.layout().addWidget(QLabel("<b>Measurement</b>"))

        self.labels_select = create_widget(annotation=Labels, label="labels_layer")

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # region properties file upload
        self.reg_props_file_widget = QWidget()
        self.reg_props_file_widget.setLayout(QVBoxLayout())
        filename_edit = FileEdit(
            mode=FileDialogMode.EXISTING_FILE, filter="*.csv", value="  "
        )
        self.reg_props_file_widget.layout().addWidget(filename_edit.native)



        # Run button
        run_button_container = QWidget()
        run_button_container.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_button_container.layout().addWidget(run_button)

        # adding all widgets to the layout
        self.layout().addWidget(title_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(self.reg_props_file_widget)
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
                self.labels_select.value,
                str(filename_edit.value.absolute())
                .replace("\\", "/")
                .replace("//", "/"),
            )

        run_button.clicked.connect(run_clicked)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            item.layout().setSpacing(0)
            item.layout().setContentsMargins(3, 3, 3, 3)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)


    # this function runs after the run button is clicked
    def run(
        self,
        labels_layer,
        reg_props_file,

    ):
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
            edited_reg_props = pd.concat([label_column, edited_reg_props], axis=1)
        
        set_features(labels_layer, edited_reg_props)
        show_table(self.viewer, labels_layer)

