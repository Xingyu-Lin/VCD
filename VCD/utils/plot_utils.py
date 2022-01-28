import numpy as np


def seg_3d_figure(data: np.ndarray, labels: np.ndarray, labelmap=None, sizes=None, fig=None):
    import plotly.colors as pc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.figure_factory as ff

    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)

    # Colormap.
    cols = np.array(pc.qualitative.Alphabet)
    labels = labels.astype(int)
    for label in np.unique(labels):
        subset = data[np.where(labels == label)]
        subset = np.squeeze(subset)
        if sizes is None:
            subset_sizes = 1.5
        else:
            subset_sizes = sizes[np.where(labels == label)]
        color = cols[label % len(cols)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker={"size": subset_sizes, "color": color, "line": {"width": 0}},
                x=subset[:, 0],
                y=subset[:, 1],
                z=subset[:, 2],
                name=legend,
            )
        )
    fig.update_layout(showlegend=True)

    # This sets the figure to be a cube centered at the center of the pointcloud, such that it fits
    # all the points.
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
            yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
            zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75),
    )
    return fig
