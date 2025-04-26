import pytest

from napari_clusters_plotter._sample_data import (
    bbbc_1_dataset,
    cells3d_curvatures,
    tgmm_mini_dataset,
)


@pytest.mark.parametrize(
    "sample_data_function",
    [bbbc_1_dataset, cells3d_curvatures, tgmm_mini_dataset],
)
def test_bbbc_1_sample_data(make_napari_viewer, sample_data_function):
    from napari.layers import Layer

    viewer = make_napari_viewer()

    sample_dataset = sample_data_function()
    for sample in sample_dataset:
        layer = Layer.create(sample[0], sample[1], sample[2])
        viewer.add_layer(layer)

    assert len(viewer.layers) == len(sample_dataset)
