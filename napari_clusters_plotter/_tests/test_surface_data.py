import napari_process_points_and_surfaces as nppas

from napari_clusters_plotter._plotter import PlotterWidget


def test_surface_data_plotting(make_napari_viewer):
    viewer = make_napari_viewer()

    gastruloid = nppas.gastruloid()

    requested_measurements = [nppas.Quality.ASPECT_RATIO, nppas.Quality.AREA]
    data_frame = nppas.surface_quality_table(gastruloid, requested_measurements)
    surface_layer = viewer.add_surface(gastruloid)
    surface_layer.properties = data_frame.to_dict(orient="list")
    surface_layer.features = data_frame

    viewer.window.add_dock_widget(PlotterWidget(viewer), area="right")

    plotter_widget = PlotterWidget(viewer)

    plotter_widget.run(
        features=data_frame,
        plot_x_axis_name="Quality.ASPECT_RATIO",
        plot_y_axis_name="Quality.AREA",
        redraw_cluster_image=True,
        force_redraw=True,
    )

    # check if plot has data
    assert plotter_widget.graphics_widget.axes.has_data()
