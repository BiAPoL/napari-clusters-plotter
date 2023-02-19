# TODO Docstrings


def unclustered_plot_parameters(
    frame_id,
    current_frame,
    n_datapoints,
):
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
    cluster_id,
    frame_id,
    current_frame,
    n_datapoints,
    color_hex_list,
):
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


def alphas_clustered(cluster_id, frame_id, current_frame, n_datapoints):
    """
    Returns a tuple of two alpha values, the first a list of alphas that depend
    on the current frame and cluster identity
    """
    initial_alpha, noise_alpha = initial_and_noise_alpha()
    alpha_f = alpha_factor(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        alphas_clustered = [
            0.3 * alpha_f * initial_alpha if id >= 0 else 0.3 * alpha_f * noise_alpha
            for id in cluster_id
        ]
        return alphas_clustered

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
    Returns a tuple of two alpha values, the first a list of alphas that depend
    on the current frame
    """
    initial_alpha, nothing = initial_and_noise_alpha()
    alpha_f = alpha_factor(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        return alpha_f * initial_alpha

    alphas_unclustered = [
        alpha_f * initial_alpha
        if tp == current_frame
        else alpha_f * initial_alpha * 0.3
        for tp in frame_id
    ]

    return alphas_unclustered


def spot_size_clustered(cluster_id, frame_id, current_frame, n_datapoints):
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
    size = gen_spot_size(n_datapoints)

    if (frame_id is None) and (current_frame is None):
        return size

    sizes = [
        size * frame_spot_factor() if tp == current_frame else size for tp in frame_id
    ]
    return sizes


def colors_clustered(cluster_id, frame_id, current_frame, color_hex_list):
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
    grey = "#9A9A9A"
    if (frame_id is None) and (current_frame is None):
        return grey

    highlight = gen_highlight()
    colors = [highlight if tp == current_frame else grey for tp in frame_id]
    return colors


# These functions generate values used by other functions
#  which determine what the visualisation looks like
def initial_and_noise_alpha():
    initial_alpha = 0.7
    noise_alpha = 0.3
    return initial_alpha, noise_alpha


def alpha_factor(n_datapoints):
    return min(1, (max(0.6, 8000 / n_datapoints)))


def frame_spot_factor():
    return 5


def gen_spot_size(n_datapoints):
    return min(10, (max(0.1, 8000 / n_datapoints))) * 2


def gen_highlight():
    return "#FFFFFF"
