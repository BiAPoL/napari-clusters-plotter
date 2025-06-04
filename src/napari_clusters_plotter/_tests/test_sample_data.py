import pytest


@pytest.mark.parametrize(
    "sample_data_function",
    ["bbbc1", "cells3d_curvatures", "tgmm_mini"],
)
def test_bbbc_1_sample_data(make_napari_viewer, sample_data_function):

    viewer = make_napari_viewer()
    viewer.open_sample("napari-clusters-plotter", sample_data_function)
    assert len(viewer.layers) > 0

    # sample_dataset = sample_data_function()
    # for sample in sample_dataset:
    #     layer = Layer.create(sample[0], sample[1], sample[2])
    #     viewer.add_layer(layer)

    # assert len(viewer.layers) == len(sample_dataset)
