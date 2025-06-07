import pytest


@pytest.mark.parametrize(
    "sample_data_function",
    ["bbbc1", "cells3d_curvatures", "tgmm_mini", "skan_skeleton"],
)
def test_bbbc_1_sample_data(make_napari_viewer, sample_data_function):

    viewer = make_napari_viewer()
    viewer.open_sample("napari-clusters-plotter", sample_data_function)
    assert len(viewer.layers) > 0
