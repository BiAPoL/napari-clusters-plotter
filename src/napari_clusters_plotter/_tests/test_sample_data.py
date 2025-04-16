def test_bbbc_1_sample_data(make_napari_viewer):
    from napari.layers import Layer

    import napari_clusters_plotter as ncp

    viewer = make_napari_viewer()

    sample_dataset = ncp.bbbc_1_dataset()
    for sample in sample_dataset:
        layer = Layer.create(sample[0], sample[1], sample[2])
        viewer.add_layer(layer)

    assert len(viewer.layers) == len(sample_dataset)


def test_tgmm_mini_dataset(make_napari_viewer):
    from napari.layers import Layer

    import napari_clusters_plotter as ncp

    viewer = make_napari_viewer()

    sample_dataset = ncp.tgmm_mini_dataset()
    for sample in sample_dataset:
        layer = Layer.create(sample[0], sample[1], sample[2])
        viewer.add_layer(layer)

    assert len(viewer.layers) == len(sample_dataset)