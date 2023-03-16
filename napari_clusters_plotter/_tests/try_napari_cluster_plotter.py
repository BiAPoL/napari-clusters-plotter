import napari
import napari_process_points_and_surfaces as nppas

viewer = napari.Viewer(ndisplay=3)

gastruloid = nppas.gastruloid()

requested_measurements = [nppas.Quality.ASPECT_RATIO, nppas.Quality.AREA]
df = nppas.surface_quality_table(gastruloid, requested_measurements)
surface_layer = viewer.add_surface(gastruloid)
surface_layer.properties = df.to_dict(orient='list')
surface_layer.features = df


