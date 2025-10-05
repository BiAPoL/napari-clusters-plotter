# Documentation

When adding a new feature to the napari-clusters-plotter, it is strongly encouraged to also add documentation to it that explains what it can do and how it can be accessed through the napari-clusters-plotter user interface. Preferrable options to do so are to create a series of screenshots or a short gif that demonstrates practical usage.

## Building the docs

Before submitting your PR, make sure the documentation pages build without error and that your changes render there as you'd expect. napari-clusters-plotter uses [Jupyter books](https://jupyterbook.org/en/stable/intro.html) for its documentation. To build, first install all necessary dependencies:

```bash
pip install -r docs/requirements.txt
```

Next, build the docs like this:

```
jupyter-book build docs/
```

And inspect the generated output under `docs/_build/html`.