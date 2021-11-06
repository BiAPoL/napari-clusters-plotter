import pandas as pd
import warnings
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QSpinBox, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
from qtpy.QtCore import QRect
from ._utilities import widgets_inactive

class ClusteringWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.current_annotation = None

        # setup layout of the whole dialog. QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())

        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())
        label_container.layout().addWidget(QLabel("<b>Clustering</b>"))

        # selection of labels layer
        choose_img_container = QWidget()
        choose_img_container.setLayout(QHBoxLayout())
        choose_img_container.layout().addWidget(QLabel("Labels layer"))
        self.label_list = QComboBox()
        self.update_label_list()
        choose_img_container.layout().addWidget(self.label_list)
        self.label_list.currentIndexChanged.connect(self.update_properties_list)

        # select properties to make a clustering from
        choose_properties_container = QWidget()
        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Measurements"))
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))
        self.update_properties_list()
        choose_properties_container.layout().addWidget(self.properties_list)

        # selection of the clustering methods
        self.clust_method_container = QWidget()
        self.clust_method_container.setLayout(QHBoxLayout())
        self.clust_method_container.layout().addWidget(QLabel("Clustering Method"))
        self.clust_method_choice_list = QComboBox()
        self.clust_method_choice_list.addItems(['   ', 'KMeans', 'HDBSCAN'])
        self.clust_method_container.layout().addWidget(self.clust_method_choice_list)

        # clustering options for KMeans
        # selection of number of clusters
        self.kmeans_settings_container = QWidget()
        self.kmeans_settings_container.setLayout(QHBoxLayout())
        self.kmeans_settings_container.layout().addWidget(QLabel("Number of Clusters"))
        self.kmeans_nr_clusters = QSpinBox()
        self.kmeans_nr_clusters.setMinimumWidth(40)
        self.kmeans_nr_clusters.setMinimum(2)
        self.kmeans_nr_clusters.setValue(2)
        self.kmeans_settings_container.layout().addWidget(self.kmeans_nr_clusters)
        self.kmeans_settings_container.setVisible(False)

        # selection of number of iterations
        self.kmeans_settings_container2 = QWidget()
        self.kmeans_settings_container2.setLayout(QHBoxLayout())
        self.kmeans_settings_container2.layout().addWidget(QLabel("Number of Iterations"))

        self.kmeans_nr_iter = QSpinBox()
        self.kmeans_nr_iter.setMinimumWidth(40)
        self.kmeans_nr_iter.setMinimum(1)
        self.kmeans_nr_iter.setMaximum(10000)
        self.kmeans_nr_iter.setValue(3000)
        self.kmeans_settings_container2.layout().addWidget(self.kmeans_nr_iter)
        self.kmeans_settings_container2.setVisible(False)

        # Clustering options for HDBSCAN
        # Todo

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        button = QPushButton("Run")

        def run_clicked():

            if self.get_selected_label() is None:
                warnings.warn("No labels image was selected!")
                return

            self.run(
                self.get_selected_label(),
                [i.text() for i in self.properties_list.selectedItems()],
                self.kmeans_nr_clusters.value(),
                self.kmeans_nr_iter.value()
            )

        button.clicked.connect(run_clicked)
        run_widget.layout().addWidget(button)

        # adding all widgets to the layout
        # side note: if widget is not added to the layout but set visible by connecting an event,
        # it opens up as a pop-up
        self.layout().addWidget(label_container)
        self.layout().addWidget(choose_img_container)
        self.layout().addWidget(choose_properties_container)
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

    def change_kmeans_clustering(self):
        widgets_inactive(self.kmeans_settings_container, self.kmeans_settings_container2,
                         active=self.clust_method_choice_list.currentText() == 'KMeans')

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

    def update_properties_list(self):
        selected_layer = self.get_selected_label()

        if selected_layer is not None:
            properties = selected_layer.properties
            if selected_layer.properties is not None:
                self.properties_list.clear()
                for p in list(properties.keys()):
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    # per default select all measurements that are not "label
                    if p != "label":
                        item.setSelected(True)


    def _on_selection(self, event=None):
        num_labels_in_viewer = len([layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

    # this function runs after the run button is clicked
    def run(self, labels_layer, selected_measurements_list, num_clusters, num_iterations):
        print("Dimensionality reduction running")
        print(labels_layer)
        print(selected_measurements_list)

        # Turn properties from layer into a dataframe
        properties = labels_layer.properties
        reg_props = pd.DataFrame(properties)

        # only select the columns the user requested
        properties_to_reduce = reg_props[selected_measurements_list]

        # reduce dimensions
        y_pred = kmeansclustering(properties_to_reduce, num_clusters, num_iterations)

        # write result back to properties
        properties["KMEANS_CLUSTER_ID"] = y_pred

        from ._utilities import show_table
        show_table(self.viewer, labels_layer)



def kmeansclustering(measurements, cluster_number, iterations):
    from sklearn.cluster import KMeans
    print('KMeans predictions started...')

    km = KMeans(n_clusters=cluster_number, max_iter=iterations, random_state=1000)

    y_pred = km.fit_predict(measurements)

    # saving prediction as a list for generating clustering image
    return y_pred


