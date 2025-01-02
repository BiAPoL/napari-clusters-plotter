import numpy as np


def test_mixed_layers(make_napari_viewer):
    from napari_clusters_plotter import PlotterWidget

    viewer = make_napari_viewer()
    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    random_image = np.random.random((5, 5))
    sample_labels = np.array(
        [
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ]
    )

    viewer.add_image(random_image)
    viewer.add_labels(sample_labels)

    #
