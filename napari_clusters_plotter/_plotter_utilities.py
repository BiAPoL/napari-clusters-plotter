import typing

import numpy as np
import pandas as pd
from PIL import ImageColor
from scipy import stats


def unclustered_plot_parameters(
    frame_id: list,
    current_frame: int,
    n_datapoints: int,
):
    """
    Returns a tuple containing alpha values, spot sizes, and colors for unclustered (noise) data points.

    Parameters
    ---------
    frame_id : list[int] or None
        A list of frames for all the data points. If None, a constant spot size and color will be returned.
    current_frame : int or None
        The current frame to highlight in the visualization.
    n_datapoints : int
        The number of data points.

    Returns
    ---------
    A tuple containing three lists: a list of alpha values, a list of spot sizes, and a list of colors.
    """
    a = alphas_unclustered(
        frame_id,
        current_frame,
        n_datapoints,
    )
    s = spot_size_unclustered(
        frame_id,
        current_frame,
        n_datapoints,
    )
    c = colors_unclustered(
        frame_id,
        current_frame,
    )
    return a, s, c


def clustered_plot_parameters(
    cluster_id: list,
    frame_id: list,
    current_frame: int,
    n_datapoints: int,
    color_hex_list: list,
):
    """
    Returns a tuple containing alpha values, spot sizes, and colors for clustered data points.

    Parameters
    __________
    cluster_id : list
        A list of cluster identities for all the data points.
    frame_id : list or None
        A list of frames for all the data points. If None, a constant spot size and color will be returned.
    current_frame : int or None
        The current frame to highlight in the visualization. If None, all frames will be the same color.
    n_datapoints : int
        The number of data points.
    color_hex_list : list
        A list of hexadecimal color codes.

    Returns
    __________
    A tuple containing three lists: a list of alpha values, a list of spot sizes, and a list of colors.
    """
    a = alphas_clustered(
        cluster_id,
        frame_id,
        current_frame,
        n_datapoints,
    )
    s = spot_size_clustered(
        cluster_id,
        frame_id,
        current_frame,
        n_datapoints,
    )
    c = colors_clustered(
        cluster_id,
        frame_id,
        current_frame,
        color_hex_list,
    )
    return a, s, c


def alphas_clustered(
    cluster_id: list, frame_id: list, current_frame: int, n_datapoints: int
):
    """
    Returns a list of alpha values for clustered data points,
    depending on whether data is a timelapse or not.

    Parameters
    -----------
    cluster_id : list
        A list of cluster identities for all the data points.
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.
    n_datapoints : int
        The number of data points.

    Returns
    ----------
    A list of alpha values for each data point.
    """
    initial_alpha, noise_alpha = initial_and_noise_alpha()
    alpha_f = alpha_factor(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        alphas_clustered = [
            0.3 * alpha_f * initial_alpha if id >= 0 else 0.3 * alpha_f * noise_alpha
            for id in cluster_id
        ]
        return alphas_clustered
    else:
        alphas_clustered = []
        for id, tp in zip(cluster_id, frame_id):
            multiplier = 0.3
            if tp == current_frame:
                multiplier = 1
            if id >= 0:
                alphas_clustered.append(multiplier * alpha_f * initial_alpha)
            else:
                alphas_clustered.append(multiplier * alpha_f * noise_alpha)
        return alphas_clustered


def alphas_unclustered(frame_id, current_frame, n_datapoints):
    """
    Returns a list of alpha values for unclustered data points.

    Parameters
    ___________
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.
    n_datapoints : int
        The number of data points.

    Returns
    ___________
    A list of alpha values for each data point.
    """
    initial_alpha, nothing = initial_and_noise_alpha()
    alpha_f = alpha_factor(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        return alpha_f * initial_alpha
    else:
        alphas_unclustered = [
            alpha_f * initial_alpha
            if tp == current_frame
            else alpha_f * initial_alpha * 0.3
            for tp in frame_id
        ]
        return alphas_unclustered


def spot_size_clustered(cluster_id, frame_id, current_frame, n_datapoints):
    """
    Calculates the size of each data point in a visualization of clusters.
    This function first generates a default size for each data point using the gen_spot_size function.
    It then adjusts the size of each point based on the cluster ID and the current frame, if provided.
    If frame_id and current_frame are None, the size of each point is determined by the cluster ID only.

    Parameters
    ----------
    cluster_id : list
        A list of cluster IDs for all the data points.
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.
    n_datapoints : int
        The total number of data points.

    Returns
    -------
    list
        A list of spot sizes for each data point.
    """
    size = gen_spot_size(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        spot_sizes = [size if id >= 0 else size / 2 for id in cluster_id]
        return spot_sizes

    spot_sizes = []
    for id, tp in zip(cluster_id, frame_id):
        multiplier = 1
        if tp == current_frame:
            multiplier = frame_spot_factor()

        if id >= 0:
            spot_sizes.append(size * multiplier)
        else:
            spot_sizes.append((size * multiplier) / 2)

    return spot_sizes


def spot_size_unclustered(frame_id, current_frame, n_datapoints):
    """
    Calculates the size of each data point in an unclustered visualization.
    This function first generates a default size for each data point using the gen_spot_size function.
    It then adjusts the size of each point based on the current frame, if provided.
    If both frame_id and current_frame are None, the default size is returned for all data points.

    Parameters
    ----------
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.
    n_datapoints : int
        The total number of data points.

    Returns
    -------
    list or int
        If both frame_id and current_frame are None, returns the default size for all data points.
        Otherwise, returns a list of spot sizes for each data point.
    """
    size = gen_spot_size(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        return size

    sizes = [
        size * frame_spot_factor() if tp == current_frame else size for tp in frame_id
    ]
    return sizes


def estimate_number_bins(data) -> int:
    """
    Estimates number of bins according Freedman–Diaconis rule

    Parameters
    ----------
    data: Numpy array

    Returns
    -------
    Estimated number of bins
    """

    est_a = (np.max(data) - np.min(data)) / (2 * stats.iqr(data) / np.cbrt(len(data)))
    return int(est_a)


def colors_clustered(cluster_id, frame_id, current_frame, color_hex_list):
    """
    Calculates the size of each data point in a clustered visualization.
    If both frame_id and current_frame are None, returns the default colors based on the cluster_id.
    Otherwise, adjusts the colors based on the current frame and highlights it.

    Parameters
    ----------
    cluster_id : list
        A list of cluster IDs for all the data points.
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.
    color_hex_list : list
        A list of hex color codes to be used for coloring the clusters.

    Returns
    -------
    list
        A list of hex color codes for each data point based on the cluster they belong to.
        The color of the data point at the current frame is highlighted.
    """
    if (frame_id is None) and (current_frame is None):
        colors = [color_hex_list[int(x) % len(color_hex_list)] for x in cluster_id]
        return colors

    highlight = gen_highlight()
    colors = [
        highlight
        if tp == current_frame
        else color_hex_list[int(x) % len(color_hex_list)]
        for x, tp in zip(cluster_id, frame_id)
    ]
    return colors


def colors_unclustered(frame_id, current_frame):
    """
    Generates a list of colors for the unclustered visualization. Firstly, the function generates a default grey color
    for all data points if both frame_id and current_frame are None. Otherwise, it returns a list of colors based
    on whether each data point's frame matches the current frame. If the frame matches, the color is determined
    by gen_highlight(), otherwise the color is the default grey.

    Parameters
    ----------
    frame_id : list or None
        A list of frames for all the data points.
    current_frame : int or None
        The current frame to highlight in the visualization.

    Returns
    -------
    list or str
        If both frame_id and current_frame are None, returns a default grey color.
        Otherwise, returns a list of colors for each data point.
    """
    grey = "#9A9A9A"
    if (frame_id is None) and (current_frame is None):
        return grey
    else:
        highlight = gen_highlight()
        colors = [highlight if tp == current_frame else grey for tp in frame_id]
        return colors


def initial_and_noise_alpha():
    """
    Returns a tuple of default values for data points opacity (visibility) for
    clustered data points and noise data points.
    """
    initial_alpha = 0.7
    noise_alpha = 0.3
    return initial_alpha, noise_alpha


def alpha_factor(n_datapoints):
    """
    Returns an alpha factor value depending on the number of datapoints in the plot.

    Parameters
    -----------
    n_datapoints : int
        The total number of data points.
    """
    return min(1, (max(0.6, 8000 / n_datapoints)))


def frame_spot_factor():
    """
    Returns a default size factor for data points.
    """
    return 5


def gen_spot_size(n_datapoints):
    """
    Generates a default size for each data point depending on the number
    of datapoints in the plot.

    Parameters
    ____________
    n_datapoints : int
        The total number of data points.
    """
    return min(10, (max(0.1, 8000 / n_datapoints))) * 2


def gen_highlight():
    """
    Returns a default color for the current timepoint visualization.
    Currently, it is color white.
    """
    return "#FFFFFF"


def get_most_frequent_cluster_id_within_feature_interval(cluster_name: str,
                                                         features: pd.DataFrame,
                                                         feature_x: str,
                                                         interval: typing.Tuple[int]) -> int:
    """Get the most frequent cluster id within a feature interval.

    Parameters
    ----------
    cluster_name : str
        cluster column name
    features : pd.DataFrame
        features dataframe
    feature_x : str
        feature x column name
    interval : typing.Tuple[int]
        tuple of (min, max) values

    Returns
    -------
    int
        the most frequent cluster id number within the interval
    """
    relevant_entries = features[[cluster_name, feature_x]]
    interval_mask = (relevant_entries[feature_x] >= interval[0]) & (relevant_entries[feature_x] <= interval[1])

    cluster_id_list = features.loc[interval_mask, cluster_name].values.tolist()
    # Efficient way of getting most frequent element in a list
    most_frequent_cluster = max(set(cluster_id_list), key=cluster_id_list.count)
    return most_frequent_cluster


def apply_cluster_colors_to_bars(
    axes: "matplotlib.axes.Axes",
    cluster_name: str,
    features: pd.DataFrame,
    number_bins: int,
    feature_x: str,
    colors: typing.List[str],
) -> "matplotlib.axes.Axes":
    """Apply cluster colors to bars in a histogram.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        the axes to which the histogram belongs
    cluster_name : str
        cluster column name
    features : pd.DataFrame
        features dataframe
    number_bins : int
        number of bins in the histogram
    feature_x : str
        feature x column name
    colors : typing.List[str]
        list of colors

    Returns
    -------
    "matplotlib.axes.Axes"
        the axes to which the histogram belongs with updated colors
    """
    assert cluster_name in features, f"Column {cluster_name} not in features."
    assert feature_x in features, f"Column {feature_x} not in features."
    # update bar colors
    for bar in axes.containers[0]:
        x_left = bar.get_x()
        x_right = x_left + bar.get_width()
        interval = (x_left, x_right)
        if bar.get_height() == 0:
            continue
        most_frequent_cluster = get_most_frequent_cluster_id_within_feature_interval(
            cluster_name=cluster_name,
            features=features,
            feature_x=feature_x,
            interval=interval)
        bar.set_color(colors[most_frequent_cluster])
        if number_bins < 100:
            bar.set_edgecolor('white')
    return axes


def make_cluster_overlay_img(
    cluster_id: str,
    features: pd.DataFrame,
    histogram_data: typing.Tuple,
    feature_x: str,
    feature_y: str,
    colors: typing.List[str],
    hide_first_cluster: bool = True,
) -> np.array:
    """
    Calculates in RGBA image of the clustering result based the results of np.histogram2d.

    Parameters
    ----------
    cluster_id : str
        Column of the clustering result
    features : pd.DataFrame
        Feature dataframe
    histogram_data : tuple
        3 element tuple with the histogram itself, x- and -y edges.
    feature_x : str
        Feature column for x-axis
    feature_y : str
        Feature column for y-axis
    colors : list
        Colors for cluster color mapping
    hide_first_cluster : bool
        Whether non-selected points are not visualized as a cluster

    Returns
    -------
    numpy array with shape (W,H,4) which represents an RGBA image.
    """

    assert cluster_id in features, f"Column {cluster_id} not in features."
    assert feature_x in features, f"Column {feature_x} not in features."
    assert feature_y in features, f"Column {feature_y} not in features."

    h, xedges, yedges = histogram_data

    relevant_entries = features[[cluster_id, feature_x, feature_y]]
    if hide_first_cluster:
        relevant_entries = features.loc[
            features[cluster_id] != features[cluster_id].min(),
            [cluster_id, feature_x, feature_y],
        ]

    cluster_overlay_rgba = np.zeros((*h.shape, 4), dtype=float)
    output_max = np.zeros(h.shape, dtype=float)

    for cluster, entries in relevant_entries.groupby(cluster_id):
        h2, _, _ = np.histogram2d(
            entries[feature_x], entries[feature_y], bins=[xedges, yedges]
        )
        mask = h2 > output_max
        np.maximum(h2, output_max, out=output_max)
        rgba = [
            float(v) / 255
            for v in list(
                ImageColor.getcolor(colors[int(cluster) % len(colors)], "RGB")
            )
        ]
        rgba.append(0.9)
        cluster_overlay_rgba[mask] = rgba

    return cluster_overlay_rgba.swapaxes(0, 1)
