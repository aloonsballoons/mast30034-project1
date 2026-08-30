"""Plotting helpers for the figures in ``main.ipynb``.

Every figure uses ``scripts/style.mplstyle``, which the notebook loads once in
its setup cell.
"""

import matplotlib.pyplot as plt


def histograms(tables, cutoffs, xlabel, title):
    """Plot binned trip counts for each service side by side.

    Args:
        tables (dict): Service name -> output of ``clean.histogram``.
        cutoffs (list of float): x values to mark with dashed lines.
        xlabel (str): x-axis label.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, axes = plt.subplots(1, len(tables), sharex=True,
                             layout="constrained")
    for ax, (service, table) in zip(axes, tables.items()):
        width = table["left"].diff().median()
        ax.bar(table["left"], table["count"], width=width, align="edge")
        for cutoff in cutoffs:
            ax.axvline(cutoff, color="black", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(service)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Trips (log scale)")
    fig.suptitle(title)
    return fig


def hourly_shares(tables, title):
    """Plot the share of trips by pickup hour, weekdays against weekends.

    Args:
        tables (dict): Service name -> output of ``clean.hourly_counts``.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, axes = plt.subplots(1, len(tables), sharey=True,
                             layout="constrained")
    for ax, (service, table) in zip(axes, tables.items()):
        for day in ["weekday", "weekend"]:
            ax.plot(table.index, table[day], marker="o", markersize=2,
                    label=day)
        ax.set_title(service)
        ax.set_xlabel("Pickup hour")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Share of trips (%)")
    axes[0].legend()
    fig.suptitle(title)
    return fig
