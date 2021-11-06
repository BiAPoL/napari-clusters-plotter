def widgets_inactive(*widgets, active):
    for widget in widgets:
        widget.setVisible(active)
