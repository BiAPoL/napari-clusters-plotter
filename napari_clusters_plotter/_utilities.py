
def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)


def show_table(viewer, labels_layer):
    from napari_skimage_regionprops import add_table
    add_table(labels_layer, viewer)


def restore_defaults(widget, defaults: dict):
    for item, val in defaults.items():
        getattr(widget, item).value = val
