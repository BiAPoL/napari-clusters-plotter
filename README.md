# napari-clusters-plotter

[![License](https://img.shields.io/pypi/l/napari-clusters-plotter.svg?color=green)](https://github.com/lazigu/napari-clusters-plotter/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-clusters-plotter.svg?color=green)](https://pypi.org/project/napari-clusters-plotter)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-clusters-plotter.svg?color=green)](https://python.org)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/napari-clusters-plotter/badges/version.svg)](https://anaconda.org/conda-forge/napari-clusters-plotter)
[![tests](https://github.com/lazigu/napari-clusters-plotter/workflows/tests/badge.svg)](https://github.com/lazigu/napari-clusters-plotter/actions)
[![codecov](https://codecov.io/gh/BiAPoL/napari-clusters-plotter/branch/main/graph/badge.svg?token=R6W2KO1NJ8)](https://codecov.io/gh/BiAPoL/napari-clusters-plotter)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-clusters-plotter.svg)](https://pypistats.org/packages/napari-clusters-plotter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-clusters-plotter)](https://www.napari-hub.org/plugins/napari-clusters-plotter)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5884658.svg)](https://doi.org/10.5281/zenodo.5884658)

A plugin to use with napari for clustering objects according to their properties.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/screencast.gif)

----------------------------------

Jump to:
- [Usage]
  - [Starting point]
  - [Measurements]
  - [Time-Lapse Measurements]
  - [Plotting]
  - [Time-Lapse Plotting]
  - [Dimensionality reduction: UMAP, t-SNE or PCA]
  - [Clustering]
  - [Plotting clustering results]
- [Installation]
- [Troubleshooting installation]
- [Contributing]
- [License]
- [Acknowledgements]

[Usage]: README.md#usage
[Starting point]: README.md#starting-point
[Measurements]: README.md#measurements
[Time-Lapse Measurements]: README.md#time-lapse-measurements
[Plotting]: README.md#plotting
[Time-Lapse Plotting]: README.md#time-lapse-plotting
[Dimensionality reduction: UMAP, t-SNE or PCA]: README.md#dimensionality-reduction-umap-t-sne-or-pca
[Clustering]: README.md#clustering
[Plotting clustering results]: README.md#plotting-clustering-results
[Installation]: README.md#installation
[Troubleshooting installation]: README.md#troubleshooting-installation
[Contributing]: README.md#contributing
[License]: README.md#license
[Acknowledgements]: README.md#acknowledgements


## Usage

### Starting point
For clustering objects according to their properties, the starting point is a [grey-value image](example_data/blobs.tif) and a label image
representing a segmentation of objects. For segmenting objects, you can for example use the
[Voronoi-Otsu-labelling approach](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes#voronoi-otsu-labelling)
in the napari plugin [napari-segment-blobs-and-things-with-membranes](https://www.napari-hub.org/plugins/napari-segment-blobs-and-things-with-membranes).

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/starting_point.png)

### Measurements
The first step is deriving measurements from the labelled image and the corresponding pixels in the grey-value image.
You can use the menu `Tools > Measurement > Measure intensity, shape and neighbor counts (ncp)` for that.
Just select the image, the corresponding label image and the measurements to analyse and click on `Run`.
A table with the measurements will open:

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/measure.png)

Afterwards, you can save and/or close the measurement table. Also, close the Measure widget.

#### Time-Lapse Measurements
If you have 3D time-lapse data this will automatically be detected. In case you have 2D time-lapse data you need to
convert it into a suitable shape using the function: `Tools > Utilities > Convert 3D stack to 2D time-lapse (time-slicer)`,
which can be found in the [napari time slicer](https://www.napari-hub.org/plugins/napari-time-slicer).
Note that tables for time-lapse data will include an additional column named "frame", which indicates which slice in
time the given row refers to. If you want to import your own csv files for time-lapse data make sure to include this column!

### Plotting

Once measurements were made, these measurements were saved in the `properties` of the labels layer which was analysed.
You can then plot these measurements using the menu `Tools > Measurement > Plot measurement (ncp)`.

In this widget, you can select the labels layer which was analysed and the measurements which should be plotted
on the X- and Y-axis. If you cannot see any options in axes selection boxes, but you have performed measurements, click
on `Update Axes/Clustering Selection Boxes` to refresh them. Click on `Run` to draw the data points in the plot area.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/plot_plain.png)

You can also manually select a region in the plot. To use lasso (freehand) tool use left mouse click, and to use a
rectangle - right click. The resulting manual clustering will also be visualized in the original image. To optimize
visualization in the image, turn off the visibility of the analysed labels layer.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/plot_interactive.png)

Hold down the SHIFT key while annotating regions in the plot to manually select multiple clusters.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/multi-select-manual-clustering.gif)

#### Time-Lapse Plotting
When you plot your time-lapse datasets you will notice that the plots look slightly different.
Datapoints of the current time frame are highlighted in white and you can see the datapoints move through the plot if you press play:

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/plotting_time-lapse_data_as_movie.gif)

You can also manually select groups using the lasso tool and plot a measurement per frame and see how the group behaves in time.
Furthermore, you could also select a group in time and see where the datapoints lie in a different feature space:

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/timelapse_manual_clustering_tips.gif)

### Dimensionality reduction: UMAP, t-SNE or PCA

For getting more insights into your data, you can reduce the dimensionality of the measurements, e.g.
using the [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/), [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
or [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
To apply it to your data use the menu `Tools > Measurement > Dimensionality reduction (ncp)`.
Select the label image that was analysed and in the list below, select all measurements that should be
dimensionality reduced. By default, all measurements are selected in the box. If you cannot see any measurements, but
you have performed them, click on `Update Measurements` to refresh the box. You can read more about parameters of both
algorithms by hovering over question marks or by clicking on them. When you are done with the selection, click on `Run`
and after a moment, the table of measurements will re-appear with two additional columns representing the reduced
dimensions of the dataset. These columns are automatically saved in the `properties` of the labels layer so there is no
need to save them for usage in other widgets unless you wish to do so.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/umap.png)

Afterwards, you can again save and/or close the table. Also, close the Dimensionality Reduction widget.

### Clustering
If manual clustering, as shown above, is not an option, you can automatically cluster your data, using these implemented algorithms:
* [k-means clustering (KMEANS)](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
* [Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
* [Gaussian Mixture Model (GMM)](https://scikit-learn.org/stable/modules/mixture.html)
* [Mean Shift (MS)](https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py)
* [Agglomerative clustering (AC)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

Therefore, click the menu `Tools > Measurement > Clustering (ncp)`,
again, select the analysed labels layer.
This time select the measurements for clustering, e.g. select _only_ the `UMAP` measurements.
Select the clustering method `KMeans` and click on `Run`.
The table of measurements will reappear with an additional column `ALGORITHM_NAME_CLUSTERING_ID` containing the cluster
ID of each datapoint.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/clustering.png)

Afterwards, you can again save and/or close the table. Also, close the clustering widget.

### Plotting clustering results
Return to the Plotter widget using the menu `Tools > Measurement > Plot measurement (ncp)`.
Select `UMAP_0` and `UMAP_1` as X- and Y-axis and the `ALGORITHM_NAME_CLUSTERING_ID` as `Clustering`, and click on `Run`.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/hdbscan_clusters_plot.png)

Example of k-means clustering results:

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/kmeans_clusters_plot.png)

## Installation

* Get a python environment, e.g. via [mini-conda](https://docs.conda.io/en/latest/miniconda.html).
  If you never used python/conda environments before, please follow the instructions
  [here](https://mpicbg-scicomp.github.io/ipf_howtoguides/guides/Python_Conda_Environments) first. It is recommended to
  install python 3.9 to your new conda environment from the start. The plugin is not yet supported with Python 3.10.
  Create a new environment, for example, like this:

```
conda create --name ncp-env python=3.9
```

* Activate the new environment via conda:

```
conda activate ncp-env
```

* Install [pyopencl](https://documen.tician.de/pyopencl/), e.g. via conda:

```
conda install -c conda-forge pyopencl
```

* Install [napari], e.g. via [pip]:

```
python -m pip install "napari[all]"
```

Afterwards, you can install `napari-clusters-plotter` via [pip]:

```
pip install napari-clusters-plotter
```

## Troubleshooting installation

- If the plugin does not appear in napari 'Plugins' menu, and in 'Plugin errors...' you can see such an error:

```
ImportError: DLL load failed while importing _cl
```

Try downloading and installing a pyopencl with a lower cl version, e.g. cl12 : pyopencl=2020.1. However, in this case,
you will need an environment with a lower python version (python=3.8).

- `Error: Could not build wheels for hdbscan which use PEP 517 and cannot be installed directly`

Install hdbscan via conda before installing the plugin:

```
conda install -c conda-forge hdbscan
```

- `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`

Similar to the above-described error, this error can occur when importing hdbscan through pip or in the wrong order. This can be fixed by installing packages separately through conda and in the following order:
```bash
conda install -c conda-forge napari pyopencl hdbscan
pip install napari-clusters-plotter
```

- `WARNING: No ICDs were found` or `LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR`

Make your system-wide implementation visible by installing either of the following conda packages:

```
conda install -c conda-forge ocl-icd-system
conda install -c conda-forge ocl_icd_wrapper_apple
```

## Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-clusters-plotter" is free and open source software

## Acknowledgements
This project was supported by the Deutsche Forschungsgemeinschaft under Germany’s Excellence Strategy – EXC2068 - Cluster of Excellence "Physics of Life" of TU Dresden.
This project has been made possible in part by grant number [2021-240341 (Napari plugin accelerator grant)](https://chanzuckerberg.com/science/programs-resources/imaging/napari/improving-image-processing/) from the Chan Zuckerberg Initiative DAF, an advised fund of the Silicon Valley Community Foundation.

## Issues

If you encounter any problems, please [file an issue](https://github.com/BiAPoL/napari-clusters-plotter/issues) along
with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[pytest]: https://docs.pytest.org/en/7.0.x/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
