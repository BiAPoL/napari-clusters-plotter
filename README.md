# napari-clusters-plotter

[![License](https://img.shields.io/pypi/l/napari-clusters-plotter.svg?color=green)](https://github.com/lazigu/napari-clusters-plotter/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-clusters-plotter.svg?color=green)](https://pypi.org/project/napari-clusters-plotter)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-clusters-plotter.svg?color=green)](https://python.org)
[![tests](https://github.com/lazigu/napari-clusters-plotter/workflows/tests/badge.svg)](https://github.com/lazigu/napari-clusters-plotter/actions)
[![codecov](https://codecov.io/gh/lazigu/napari-clusters-plotter/branch/master/graph/badge.svg)](https://codecov.io/gh/lazigu/napari-clusters-plotter)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-clusters-plotter.svg)](https://pypistats.org/packages/napari-clusters-plotter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-clusters-plotter)](https://www.napari-hub.org/plugins/napari-clusters-plotter)

A plugin to use with napari for clustering objects according to their properties.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/screencast.gif)

## Usage

### Starting point
For clustering objects according to their properties, the starting point is a [grey-value image](example_data/blobs.tif) and a label image
representing a segmentation of objects. For segmenting objects, you can for example use the 
[Voronoi-Otsu-labeling approach](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes#voronoi-otsu-labeling)
in the napari plugin [napari-segment-blobs-and-things-with-membranes](https://www.napari-hub.org/plugins/napari-segment-blobs-and-things-with-membranes).

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/starting_point.png)

### Measurements
The first step is deriving measurements from the labeled image and the corresponding pixels in the grey-value image.
You can use the menu `Tools > Measurement > Measure intensity, shape and neighbor counts (ncp)` for that. 
Just select the image, the corresponding label image and the measurements to analyse and click on `Run`.
A table with the measurements will open:

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/measure.png)

Afterwards, you can save and/or close the measurement table. Also, close the Measure widget. Or if you want you can
interact with labels and see which row of the table corresponds to which labelled object. For this, use the Pick mode
in napari and activate the show selected checkbox. Alternatively, you can also select a specific row of the table and
appropriate label is displayed (make sure that `show selected` checkbox is selected).


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

You can also select a labeled object in the original labels layer (not "cluster_ids_in_space" layer) using the `Pick`
mode in napari and see which data point in the plot it corresponds to. Again, make sure that `show selected` checkbox is
selected.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/select_in_layer.png)


### Dimensionality reduction: UMAP or t-SNE

For getting more insights into your data, you can reduce the dimensionality of the measurements, e.g.
using the [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/) or [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
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

### Clustering: k-means or HDBSCAN
If manual clustering, as shown above, is not an option, you can automatically cluster your data, e.g. using the
[k-means clustering algorithm](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
or [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).
Therefore, click the menu `Tools > Measurement > Clustering (ncp)`,
again, select the analysed labels layer.
This time select the measurements for clustering, e.g. select _only_ the `UMAP` measurements.
Select the clustering method `KMeans` and click on `Run`. 
The table of measurements will reappear with an additional column `ALGORITHM_NAME_CLUSTERING_ID` containing the cluster
ID of each datapoint.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/kmeans_clustering.png)

Afterwards, you can again save and/or close the table. Also, close the clustering widget.

### Plotting clustering results
Return to the Plotter widget using the menu `Tools > Measurement > Plot measurement (ncp)`.
Select `UMAP_0` and `UMAP_1` as X- and Y-axis and the `ALGORITHM_NAME_CLUSTERING_ID` as `Cluster`, and click on `Run`.

![](https://github.com/BiAPoL/napari-clusters-plotter/raw/main/images/plot_umap.png)

## Installation

* Get a python environment, e.g. via [mini-conda](https://docs.conda.io/en/latest/miniconda.html). 
  If you never used python/conda environments before, please follow the instructions 
  [here](https://mpicbg-scicomp.github.io/ipf_howtoguides/guides/Python_Conda_Environments) first. It is recommended to
  install python 3.9 to your new conda environment from the start. The plugin is not yet supported with Python 3.10.
  Create a new environment, for example, like this:

```
conda create --name napari-clusters-plotter python=3.9
```

* Activate the new environment and install [pyopencl](https://documen.tician.de/pyopencl/), e.g. via conda:

```
conda install -c conda-forge pyopencl
```

* Install [napari], e.g. via [pip]:

```
pip install "napari[all]"
```

Afterwards, you can install `napari-clusters-plotter` via [pip]:

    pip install napari-clusters-plotter

## Troubleshooting installation

If the plugin does not appear in napari 'Plugins' menu, and in 'Plugin errors...' you can see such an error:

```
ImportError: DLL load failed while importing _cl
```

Try downloading and installing a pyopencl with a lower cl version, e.g. cl12 : pyopencl=2020.1. However, in this case,
you will need an environment with a lower python version (python=3.8).

If you encounter such an error:

```
Error: Could not build wheels for hdbscan which use PEP 517 and cannot be installed directly
```

You can install hdbscan via conda before installing the plugin:

```
conda install -c conda-forge hdbscan
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-clusters-plotter" is free and open source software

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
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
