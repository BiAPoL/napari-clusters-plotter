import napari
import napari_process_points_and_surfaces as nppas
from napari_clusters_plotter._plotter import PlotterWidget


viewer = napari.Viewer(ndisplay=3)

gastruloid = nppas.gastruloid()

requested_measurements = [nppas.Quality.ASPECT_RATIO, nppas.Quality.AREA]
data_frame = nppas.surface_quality_table(gastruloid, requested_measurements)
surface_layer = viewer.add_surface(gastruloid)
surface_layer.properties = data_frame.to_dict(orient='list')
surface_layer.features = data_frame

viewer.window.add_dock_widget(PlotterWidget(viewer), area='right')

napari.run()