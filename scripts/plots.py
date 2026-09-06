"""Plotting helpers for the figures and tables in ``main.ipynb``.

Every figure uses ``scripts/style.mplstyle``, which the notebook loads once in
its setup cell.
"""

import re

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML
from matplotlib import cm, colors

from scripts import config

# Service names on every figure and table
SERVICE_NAMES = {"yellow": "Yellow Taxi",
                 "fhvhv": "High Volume For-Hire Vehicle"}


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
        ax.set_title(SERVICE_NAMES[service])
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
        ax.set_title(SERVICE_NAMES[service])
        ax.set_xlabel("Pickup hour")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Share of trips (%)")
    axes[0].legend()
    fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------
# Section 5: exploration and maps
# ---------------------------------------------------------------------------
GROUP_NAMES = {"treated": "Treated (touches CBD)", "ring": "Ring",
               "control": "Control"}
ZONE_GROUP_NAMES = {"cbd": "CBD", "ring": "Ring", "control": "Control"}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BAND_NAMES = {"overnight": "Overnight\n21:00-05:00",
              "morning_peak": "Morning\n05:00-10:00",
              "midday": "Midday\n10:00-16:00",
              "evening": "Evening\n16:00-21:00"}
# Matrices between linear sRGB and OKLab (Ottosson 2020), a colour space
# where equal distances look like equal differences in colour
_RGB_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                        [0.2119034982, 0.6806995451, 0.1073969566],
                        [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_TO_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                          [1.9779984951, -2.4285922050, 0.4505937099],
                          [0.0259040371, 0.7827717662, -0.8086757660]])


def _oklab_to_srgb(lab):
    """Convert OKLab colours (rows of L, a, b) to sRGB values in [0, 1]."""
    lms = (lab @ np.linalg.inv(_LMS_TO_OKLAB).T) ** 3
    linear = np.clip(lms @ np.linalg.inv(_RGB_TO_LMS).T, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def _diverging_colormap(n=256):
    """A perceptually uniform blue - light grey - red colour map.

    Each half runs from light grey (OKLab lightness 0.96) to a dark pole
    (lightness 0.40) of a fixed hue, blue for the low end and red for the
    high end. Lightness falls linearly; chroma grows towards the pole with a
    bulge in the middle, so mid-range values stay clearly blue or red rather
    than greyish. The path is then resampled at equal distances in OKLab, so
    every step along the scale is an equally visible change in colour, and
    both halves are the same length, so +x% and -x% look equally strong.
    ``RdBu_r``, used before, has steps up to 5 times larger in some parts
    of the scale than in others.

    Args:
        n (int): Number of colours.

    Returns:
        matplotlib.colors.ListedColormap: The colour map.
    """
    u = np.linspace(0, 1, 2001)  # 0 = centre, 1 = pole
    lightness = 0.96 - 0.56 * u
    chroma = 0.16 * u + 0.09 * np.sin(np.pi * u) * np.sqrt(u)
    halves = []
    for hue in [262, 27]:  # blue, red (degrees in OKLab)
        path = np.column_stack([lightness,
                                chroma * np.cos(np.radians(hue)),
                                chroma * np.sin(np.radians(hue))])
        # Distance along the path, then colours at equal distances
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0),
                                                     axis=1))]
        equal = np.linspace(0, distance[-1], n // 2)
        halves.append(np.column_stack([np.interp(equal, distance, path[:, k])
                                       for k in range(3)]))
    lab = np.vstack([halves[0][::-1], halves[1]])
    return colors.ListedColormap(_oklab_to_srgb(lab), name="blue_grey_red")


# Diverging scale for every map and heatmap of change: blue = fall,
# light grey = no change, red = rise. Perceptually uniform (see
# ``_diverging_colormap``); blue against red also stays distinguishable
# with red-green colour blindness
DIVERGING = _diverging_colormap()
# Fill for zones greyed out because they have too few trips
TOO_FEW_COLOUR = "#d4d3cd"


def save(fig, name):
    """Save a figure to ``plots/`` as a PDF (the style file's format).

    Args:
        fig (matplotlib.figure.Figure): The figure.
        name (str): File name without the extension.

    Returns:
        pathlib.Path: Where the figure was saved.
    """
    path = config.PLOTS_DIR / f"{name}.pdf"
    fig.savefig(path)
    return path


# Labels for the column names and code values that ``heading`` can't work
# out from the words alone
LABELS = {
    # Services, companies, groups, time bands and datasets, as columns or
    # values
    **SERVICE_NAMES,
    "HV0003": "Uber", "HV0005": "Lyft",
    "cbd": "CBD", "ring": "Ring", "control": "Control", "treated": "Treated",
    "ring_adjacent": "Ring, adjacent", "ring_across": "Ring, across water",
    "control_near": "Control, near", "control_far": "Control, far",
    "all": "All", "overnight": "Overnight", "midday": "Midday",
    "evening": "Evening",
    "lookup": "Zone lookup", "shapefile": "Zone shapefile",
    "weather": "Weather",
    # Cleaning steps and rules
    "raw": "Raw", "dropoff_after_pickup": "Drop-off after pickup",
    "known_ratecode": "Known rate code", "all rules": "All rules",
    # Statistics from ``describe``
    "count": "Count", "mean": "Mean", "std": "Standard deviation",
    "min": "Min", "max": "Max",
    # Group columns
    "zone_group_fine": "Fine zone group", "trip_group_coarse": "Trip group",
    # TLC columns
    "LocationID": "Zone ID", "PULocationID": "Pickup zone ID",
    "VendorID": "Vendor", "RatecodeID": "Rate code",
    "passenger_count": "Passengers", "hvfhs_license_num": "Company",
    "fare_amount": "Fare ($)", "driver_pay": "Driver pay ($)",
    "trip_distance": "Trip distance (mi)", "trip_miles": "Trip distance (mi)",
    "trip_minutes": "Trip time (min)", "extra": "Extra ($)",
    "driver_extra": "Driver surcharge ($)", "fee": "CBD fee ($)",
    # Columns made in the project
    "area_sq_mi": "Area (sq mi)",
    "min_columns": "Fewest columns", "max_columns": "Most columns",
    "temp_max_c": "Max temperature (°C)",
    "temp_min_c": "Min temperature (°C)",
    "null_pct": "Rows with a null (%)",
    "flex_fare": "Flex Fare trips", "flex_pct": "Flex Fare (%)",
    "negative": "Negative fares", "zero": "Zero fares",
    "negative_pct": "Negative fares (%)",
    "reversal_pct": "Reversals (% of negative fares)",
    "matched_pct": "With a surcharge (%)",
    "no_fallback_pct": "With a surcharge, no CBD fallback (%)",
    "trips_pct": "Share of 2025 trips (%)", "hours": "Pickup hours seen",
    "zero_time": "Drop-off at or before pickup",
    "share_of_yellow": "Share of yellow rows (%)",
    "vendor_7": "Vendor 7 (%)", "all_yellow": "All yellow (%)",
    "daily_before": "Pickups a day, before",
    "eph_before": "Earnings per hour, before ($)",
    "eph_after": "Earnings per hour, after ($)",
    "eph_change_pct": "Earnings per hour, change (%)",
    "p90_wait_min": "90th percentile wait (min)",
}
# Last words of a column name that are units, shown in brackets
UNITS = {"pct": "%", "ft": "ft", "mm": "mm", "mph": "mph", "min": "min",
         "gb": "GB", "mb": "MB"}
ACRONYMS = {"cbd": "CBD", "id": "ID", "nyc": "NYC"}


def heading(name):
    """Turn a column name into a table heading.

    Names in ``LABELS`` use the label given there. Any other name has its
    underscores turned into spaces, a unit at the end put in brackets and its
    first letter capitalised, so ``distance_to_cbd_ft`` becomes "Distance to
    CBD (ft)". Names that aren't strings, such as years, are left alone.

    Args:
        name: A column name.

    Returns:
        The heading.
    """
    if not isinstance(name, str) or not name:
        return name
    if name in LABELS:
        return LABELS[name]
    words = name.split("_")
    unit = UNITS.get(words[-1].lower()) if len(words) > 1 else None
    if unit:
        words = words[:-1]
    text = " ".join(ACRONYMS.get(word.lower(), word) for word in words)
    text = text[0].upper() + text[1:]
    return f"{text} ({unit})" if unit else text


def value_label(value):
    """Label a code value in a table cell, such as ``fhvhv``.

    Only values in ``LABELS`` and lowercase names with underscores, such as
    ``ring_adjacent``, are code. Anything else, such as a zone name or a
    number, is returned unchanged.

    Args:
        value: A cell value.

    Returns:
        The label, or ``value`` itself.
    """
    if isinstance(value, str) and (value in LABELS
                                   or re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)+",
                                                   value)):
        return heading(value)
    return value


# Centred headings, and a line between every two columns. Hiding the
# table's own border removes the lines on its outer edges
_EDGES = ("border-left: 1px solid var(--jp-border-color1, "
          "rgba(128, 128, 128, 0.5));")
TABLE_STYLES = [
    {"selector": "",
     "props": "border-collapse: collapse; border-style: hidden;"},
    {"selector": "th", "props": "text-align: center; " + _EDGES},
    {"selector": "td", "props": _EDGES},
]


def flat(table, merge=None):
    """Style a table for the notebook, with every heading on one row.

    pandas shows a named index on its own row under the column headings, and
    the name of the columns in the top-left corner. This turns a named index
    into ordinary columns (an unnamed one, such as 0, 1, 2, ..., is dropped),
    hides the row numbers, centres the headings and draws a line between
    every two columns. Column names become readable headings (``heading``)
    and code values in the cells become labels (``value_label``).

    Floats are shown with the fewest decimals (up to 6) that every float in
    the table needs, so a table rounded with ``.round(1)`` shows one decimal.

    Args:
        table (pandas.DataFrame): Any table.
        merge (list of str, optional): Columns whose repeated values are
            shown once, in one cell spanning the rows. They move to the left
            of the table in this order, and each is merged only within one
            cell of the column before it. Sort the table by them first.

    Returns:
        pandas.io.formats.style.Styler: The table, ready to display, or
        IPython.display.HTML when ``merge`` is given.
    """
    table = table.reset_index(drop=not any(table.index.names))
    table = table.rename_axis(columns=[None] * table.columns.nlevels)
    table = table.rename(columns=heading)
    text = table.select_dtypes(["object", "string", "category"]).columns
    if len(text):
        table[text] = table[text].astype(object).map(value_label)
    merge = [heading(name) for name in merge or []]
    floats = table.select_dtypes("float").to_numpy().ravel()
    floats = floats[~np.isnan(floats)]
    decimals = next((d for d in range(6) if (floats.round(d) == floats).all()),
                    6)
    dates = [column for column in table.columns
             if pd.api.types.is_datetime64_any_dtype(table[column])
             and (table[column].dt.normalize() == table[column]).all()]
    if not merge:
        styled = (table.style.hide(axis="index")
                  .format(precision=decimals, na_rep="NaN")
                  .set_table_styles(TABLE_STYLES))
        # Dates at midnight show without the time
        return (styled.format("{:%Y-%m-%d}", subset=dates) if dates
                else styled)

    # pandas only merges repeated index labels, and never the last index
    # level, so the merged columns become the index above a hidden level of
    # row numbers
    dates = [column for column in dates if column not in merge]
    # Jupyter shades every other row, which gives a merged cell the shade of
    # its first row only. Shade every other group of rows instead, numbering
    # the groups by the runs of the first merged column
    first = table[merge[0]]
    shaded = ((first != first.shift()).cumsum() % 2 == 0).to_numpy()
    shade = "background: var(--jp-rendermime-table-row-background, " \
            "rgba(128, 128, 128, 0.1));"
    table = table.set_index(merge + [pd.RangeIndex(len(table))])
    styled = (table.style
              .hide(axis="index", names=True)
              .hide(axis="index", level=len(merge))
              .format(precision=decimals, na_rep="NaN")
              .apply(lambda column: np.where(shaded, shade, ""))
              .apply_index(lambda labels: np.where(shaded, shade, ""),
                           level=list(range(len(merge))))
              .set_table_styles(TABLE_STYLES + [
                  {"selector": "th.row_heading",
                   "props": "font-weight: normal; vertical-align: middle;"},
                  {"selector": "tbody tr", "props": "background: none;"}]))
    if dates:
        styled = styled.format("{:%Y-%m-%d}", subset=dates)
    # The merged columns' headings go in the empty top-left cells, so every
    # heading stays on one row
    html = styled.to_html()
    for name in merge:
        html = html.replace("&nbsp;</th>", f"{name}</th>", 1)
    return HTML(html)


def relative_trips(table, toll_date):
    """Figure 1: monthly trips relative to control, one panel per service.

    Args:
        table (pandas.DataFrame): Output of ``explore.relative_trips``.
        toll_date (datetime.date): Where to draw the toll line.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    services = [s for s in SERVICE_NAMES if s in set(table["service"])]
    fig, axes = plt.subplots(1, len(services), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for ax, service in zip(axes[0], services):
        rows = table[table["service"] == service]
        for group in ["treated", "ring"]:
            ax.plot(rows["month"], rows[group], marker="o", markersize=3,
                    label=GROUP_NAMES[group])
        ax.axvline(pd.Timestamp(toll_date), color="black", linestyle="--",
                   linewidth=0.8)
        ax.axhline(100, color="#8a8983", linewidth=0.6)
        ax.set_title(SERVICE_NAMES[service])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0][0].set_ylabel("Trips relative to control\n(2024 average = 100)")
    axes[0][0].annotate("Toll starts", xy=(pd.Timestamp(toll_date), 1),
                        xycoords=("data", "axes fraction"),
                        xytext=(-4, -2), textcoords="offset points",
                        ha="right", va="top", fontsize=10)
    fig.legend(*axes[0][0].get_legend_handles_labels(),
               loc="outside lower center", ncols=2)
    return fig


def _cbd_outline(shapes, labels):
    """The CBD boundary, as one shape.

    The zone polygons don't meet exactly, so their union has hairline gaps
    inside it. Growing and then shrinking the shape by 50 ft closes them,
    leaving only the outer boundary.
    """
    cbd = labels.loc[labels["zone_group"] == "cbd", "LocationID"]
    union = shapes[shapes["LocationID"].isin(cbd)].geometry.union_all()
    return union.buffer(50).buffer(-50)


def zone_maps(shapes, labels, table, columns, titles, limits, too_few_label):
    """Maps of change by pickup zone, side by side, with the CBD outlined.

    Zones where ``table["enough_trips"]`` is false are filled grey and
    hatched instead of coloured.

    Args:
        shapes (geopandas.GeoDataFrame): Output of
            ``zones.load_zone_shapes``.
        labels (pandas.DataFrame): Zone labels.
        table (pandas.DataFrame): One service's rows of
            ``explore.zone_change``.
        columns (list of str): Column to map in each panel (values in %).
        titles (list of str): Panel titles.
        limits (list of float): Colour scale limit for each panel; the
            scale runs from -limit to +limit, and values past it get the
            end colour.
        too_few_label (str): Legend text for the grey zones.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    data = shapes.merge(table, left_on="LocationID", right_on="PULocationID",
                        how="left")
    enough = data["enough_trips"].fillna(False).astype(bool)
    outline = gpd.GeoSeries([_cbd_outline(shapes, labels)], crs=shapes.crs)
    # Inset extent: the CBD and the ring, with a small margin
    near = labels.loc[labels["zone_group"].isin(["cbd", "ring"]),
                      "LocationID"]
    zoom = shapes[shapes["LocationID"].isin(near)].total_bounds
    zoom = zoom + np.array([-1, -1, 1, 1]) * 2000
    fig, axes = plt.subplots(1, len(columns), figsize=(6.5, 3.9),
                             layout="constrained", squeeze=False)
    for ax, column, title, limit in zip(axes[0], columns, titles, limits):
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        shown = enough & data[column].notna()
        # The CBD is small at city scale, so an inset zooms in on it and
        # the ring (top left, over New Jersey, where the map is empty)
        inset = ax.inset_axes([-0.04, 0.42, 0.56, 0.58])
        for target in [ax, inset]:
            data[shown].plot(column=column, cmap=DIVERGING, norm=norm,
                             ax=target, edgecolor="white", linewidth=0.2)
            data[~shown].plot(ax=target, color=TOO_FEW_COLOUR,
                              edgecolor="white", linewidth=0.2,
                              hatch="////")
            outline.boundary.plot(ax=target, color="black", linewidth=1.0)
            target.set_xticks([])
            target.set_yticks([])
        inset.set_xlim(zoom[0], zoom[2])
        inset.set_ylim(zoom[1], zoom[3])
        inset.set_aspect("equal")
        inset.grid(False)
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#52514e")
        ax.indicate_inset_zoom(inset, edgecolor="#52514e", alpha=1)
        ax.set_axis_off()
        ax.set_title(title)
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=DIVERGING),
                           ax=ax, orientation="horizontal", shrink=0.8,
                           extend="both", pad=0.01)
        bar.set_label("Change, 2025 vs 2024 (%)")
        bar.outline.set_visible(False)
    handles = [
        mpatches.Patch(facecolor=TOO_FEW_COLOUR, hatch="////",
                       edgecolor="white", label=too_few_label),
        mlines.Line2D([], [], color="black", linewidth=1.0,
                      label="Congestion relief zone (CBD)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    return fig


def _text_colour(background):
    """White or near-black text, whichever contrasts more with a colour.

    Contrast is the WCAG ratio of relative luminances, so the choice follows
    how light the colour looks rather than its RGB values.
    """
    def luminance(colour):
        rgb = np.array(colors.to_rgb(colour))
        linear = np.where(rgb <= 0.04045, rgb / 12.92,
                          ((rgb + 0.055) / 1.055) ** 2.4)
        return linear @ [0.2126, 0.7152, 0.0722]

    shade = luminance(background)
    on_white = 1.05 / (shade + 0.05)
    on_black = (shade + 0.05) / (luminance("#0b0b0b") + 0.05)
    return "white" if on_white > on_black else "#0b0b0b"


def band_day_heatmaps(tables, titles, limit, label):
    """Heatmaps of change by time band (rows) and day of week (columns).

    Args:
        tables (list of pandas.DataFrame): Outputs of
            ``explore.relative_change`` or ``explore.band_day_table``.
        titles (list of str): Panel titles.
        limit (float): Colour scale runs from -limit to +limit (%).
        label (str): Colour bar label.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5, 2.8),
                             sharey=True, layout="constrained",
                             squeeze=False)
    for ax, table, title in zip(axes[0], tables, titles):
        values = table.to_numpy(dtype=float)
        ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            if np.isnan(value):
                continue
            ax.text(col, row, f"{value:+.0f}", ha="center", va="center",
                    fontsize=10,
                    color=_text_colour(DIVERGING(norm(value))))
        ax.set_xticks(range(table.shape[1]),
                      [DAY_NAMES[d - 1] for d in table.columns])
        ax.set_yticks(range(table.shape[0]),
                      [BAND_NAMES.get(b, b) for b in table.index])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=DIVERGING),
                       ax=axes[0].tolist(), extend="both", shrink=0.9)
    bar.set_label(label)
    bar.outline.set_visible(False)
    return fig


def before_after_histograms(before, after, xlabels):
    """Binned counts before and after cleaning, one row per service.

    Args:
        before (dict): Service -> {column: output of
            ``explore.histograms``} on the raw data.
        after (dict): The same on the cleaned data.
        xlabels (dict): Column -> x-axis label. Columns are plotted in
            this order.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    services = list(before)
    fig, axes = plt.subplots(len(services), len(xlabels),
                             figsize=(6.5, 2.2 * len(services)),
                             layout="constrained", squeeze=False)
    for row, service in enumerate(services):
        for col, (column, xlabel) in enumerate(xlabels.items()):
            ax = axes[row][col]
            for tables, name in [(before, "raw"), (after, "cleaned")]:
                table = tables[service][column]
                width = table["left"].diff().median()
                ax.stairs(table["count"], np.append(
                    table["left"], table["left"].iloc[-1] + width),
                    label=name, linewidth=1.0)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(f"{SERVICE_NAMES[service]}\ntrips (log)")
    axes[0][0].legend()
    return fig


def monthly_lines(table, ylabel, title, toll_date=None):
    """One line per column of a monthly table.

    Args:
        table (pandas.DataFrame): Rows are months (dates or periods),
            columns are series.
        ylabel (str): y-axis label.
        title (str): Figure title.
        toll_date (datetime.date, optional): Draw the toll line here.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, ax = plt.subplots(layout="constrained")
    index = table.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    for column in table.columns:
        ax.plot(index, table[column], marker="o", markersize=3,
                label=GROUP_NAMES.get(column,
                                      ZONE_GROUP_NAMES.get(column, column)))
    if toll_date is not None:
        ax.axvline(pd.Timestamp(toll_date), color="black", linestyle="--",
                   linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig
