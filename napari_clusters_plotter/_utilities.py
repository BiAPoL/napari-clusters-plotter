
def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


def show_table(viewer, labels_layer):
    from napari_skimage_regionprops import add_table
    add_table(labels_layer, viewer)


def restore_defaults(widget, defaults: dict):
    for item, val in defaults.items():
        getattr(widget, item).value = val


def set_features(layer, tabular_data):
    if hasattr(layer, "properties"):
        layer.properties = tabular_data
    if hasattr(layer, "features"):
        layer.features = tabular_data


def get_layer_tabular_data(layer):
    if hasattr(layer, "features") and layer.features is not None:
        return layer.features
    if hasattr(layer, "properties") and layer.properties is not None:
        return layer.properties
    return None
