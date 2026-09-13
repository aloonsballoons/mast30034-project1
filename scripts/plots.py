"""Plotting helpers for the figures and tables in ``main.ipynb``.

Every figure uses ``scripts/style.mplstyle``, which the notebook loads once in
its setup cell.
"""

import re

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
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
               "control": "Control", "ring_adjacent": "Ring, touching CBD",
               "ring_across": "Ring, across water"}
ZONE_COLOURS = {"cbd": "#2a78d6", "ring": "#eb6834",
                "control": "#9a998f"}
ZONE_GROUP_NAMES = {"cbd": "Tolled zone", "ring": "Ring",
                    "control": "Control"}
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
# One-signed panels use half of it as a sequential scale, so the whole
# ramp carries the signal instead of only one side of it
SEQUENTIAL_FALL = colors.ListedColormap(
    DIVERGING(np.linspace(0, 0.5, 128)), name="blue_fall")
SEQUENTIAL_RISE = colors.ListedColormap(
    DIVERGING(np.linspace(0.5, 1, 128)), name="red_rise")
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
    "shared_pct": "Shared rides (%)",
    "eph_known_pct": "Earnings per hour known (%)",
    "median_speed_mph": "Median speed (mph)",
    "median_trip_miles": "Median distance (mi)",
    "median_trip_minutes": "Median trip time (min)",
    "median_earnings_per_engaged_hour": "Median earnings per hour ($)",
    "pickup_group": "Pickup group", "measure": "Measure",
    # Section 6: models
    "trips": "Trips", "log_eph": "Earnings per engaged hour",
    "ci_low_pct": "95% CI low (%)", "ci_high_pct": "95% CI high (%)",
    "usual": "Usual level", "year": "Year", "month": "Month",
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
    fig, axes = plt.subplots(1, len(services), figsize=(6.5, 2.9),
                             sharex=True, sharey=True,
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


def _annotate_zones(ax, data, offsets, fontsize=8.5):
    """Name zones on a map, each with a leader line to its own shape.

    The text sits on top of the choropleth, so it is drawn with a white
    outline to stay readable over a dark fill.

    A few taxi zones share a name (``zones.shared_names``), and this takes
    the first shape of any that do. None of the zones named in the report
    is one of them.

    Args:
        ax (matplotlib.axes.Axes): The panel to draw on.
        data (geopandas.GeoDataFrame): Zones, with a ``zone`` name column.
        offsets (dict): Zone name -> (dx, dy) label offset in points from
            the zone. The sign of dx and dy also picks the alignment, so
            the text always runs away from its zone.
        fontsize (float): Label size in points.
    """
    for name, (dx, dy) in offsets.items():
        match = data[data["zone"] == name]
        if match.empty:
            continue
        # A centroid can fall outside a zone that is L-shaped or split
        # across islands, so the leader line starts from a point the
        # shape is guaranteed to contain
        point = match.geometry.iloc[0].representative_point()
        text = ax.annotate(
            name, xy=(point.x, point.y), xytext=(dx, dy),
            textcoords="offset points", fontsize=fontsize, color="#0b0b0b",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            arrowprops={"arrowstyle": "-", "linewidth": 0.7,
                        "color": "#52514e", "shrinkA": 1, "shrinkB": 1})
        text.set_path_effects([patheffects.withStroke(linewidth=2.4,
                                                      foreground="white")])


def zone_maps(shapes, labels, table, columns, titles, limits, too_few_label,
              city_labels=None, zoom_labels=None):
    """Map of change by pickup zone: the whole city, then the CBD up close.

    Each column gets two panels. The 38 CBD zones cover about 1% of the
    city's area, so at city scale they are a few pixels each and the panel
    can only show the broad pattern; panel (b) zooms to the CBD and its
    ring, which is where the report reads individual zones. Zones where
    ``table["enough_trips"]`` is false are filled grey and hatched instead
    of coloured.

    Args:
        shapes (geopandas.GeoDataFrame): Output of
            ``zones.load_zone_shapes``.
        labels (pandas.DataFrame): Zone labels.
        table (pandas.DataFrame): One service's rows of
            ``explore.zone_change``.
        columns (list of str): Column to map in each pair of panels
            (values in %).
        titles (list of str): Title for each column's first panel.
        limits (list of float): Colour scale limit for each column; the
            scale runs from -limit to +limit, and values past it get the
            end colour.
        too_few_label (str): Legend text for the grey zones.
        city_labels (dict, optional): Zone name -> (dx, dy) offset in
            points, for the zones to name on the city panel.
        zoom_labels (dict, optional): The same for the zoom panel.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    data = shapes.merge(table, left_on="LocationID", right_on="PULocationID",
                        how="left")
    enough = data["enough_trips"].fillna(False).astype(bool)
    outline = gpd.GeoSeries([_cbd_outline(shapes, labels)], crs=shapes.crs)
    # Zoom extent: the CBD and the ring, with a small margin
    near = labels.loc[labels["zone_group"].isin(["cbd", "ring"]),
                      "LocationID"]
    zoom = shapes[shapes["LocationID"].isin(near)].total_bounds
    zoom = zoom + np.array([-1, -1, 1, 1]) * 2000
    # Two panels per column, each the width of a single-panel figure, so
    # the figure fills the report's text width and its fonts stay the size
    # of every other figure's
    # Both maps are about as tall as they are wide once the aspect is
    # equal, so the height is set from the panel width, not the other way
    fig, axes = plt.subplots(1, 2 * len(columns),
                             figsize=(6.5 * len(columns), 4.6),
                             layout="constrained", squeeze=False)
    panels = axes[0].reshape(len(columns), 2)
    for (city, close), column, title, limit in zip(panels, columns, titles,
                                                   limits):
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        shown = enough & data[column].notna()
        for target, width in [(city, 0.2), (close, 0.4)]:
            data[shown].plot(column=column, cmap=DIVERGING, norm=norm,
                             ax=target, edgecolor="white", linewidth=width)
            data[~shown].plot(ax=target, color=TOO_FEW_COLOUR,
                              edgecolor="white", linewidth=width,
                              hatch="////")
            outline.boundary.plot(ax=target, color="black", linewidth=1.0)
            target.set_xticks([])
            target.set_yticks([])
            target.set_axis_off()
        # The box on the city panel is the area panel (b) enlarges
        city.add_patch(mpatches.Rectangle(
            (zoom[0], zoom[1]), zoom[2] - zoom[0], zoom[3] - zoom[1],
            fill=False, edgecolor="#52514e", linewidth=0.8))
        close.set_xlim(zoom[0], zoom[2])
        close.set_ylim(zoom[1], zoom[3])
        close.set_aspect("equal")
        _annotate_zones(city, data, city_labels or {})
        _annotate_zones(close, data, zoom_labels or {})
        city.set_title(f"(a) {title}")
        close.set_title("(b) The tolled zone up close")
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=DIVERGING),
                           ax=[city, close], orientation="horizontal",
                           shrink=0.55, extend="both", pad=0.01)
        bar.set_label("Change, 2025 vs 2024 (%)")
        bar.outline.set_visible(False)
    handles = [
        mpatches.Patch(facecolor=TOO_FEW_COLOUR, hatch="////",
                       edgecolor="white", label=too_few_label),
        mlines.Line2D([], [], color="black", linewidth=1.0,
                      label="Congestion relief zone (CBD)"),
    ]
    fig.legend(handles=handles, loc="outside lower center",
               ncols=2 * len(columns))
    return fig


def zone_scatter(table, labels, annotate=None, sizes=(4, 70)):
    """Each zone's change in pickups against its change in earnings.

    The two maps show one change each, so the relationship between them has
    to be read across the pair. Here both are on one pair of axes, with the
    zones that touch the tolled area coloured, which is the comparison the
    text makes.

    Args:
        table (pandas.DataFrame): One service's rows of
            ``explore.zone_change``. Only rows with ``enough_trips`` are
            drawn.
        labels (pandas.DataFrame): Zone labels, for each zone's name and
            group.
        annotate (dict, optional): Zone name -> (dx, dy) offset in points
            for the zones to label.
        sizes (tuple): Marker area for the smallest and largest zone, so
            the busy zones read as the heavy ones.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    named = labels.set_index("LocationID")[["Zone", "zone_group"]]
    rows = (table[table["enough_trips"]].join(named, on="PULocationID")
            .dropna(subset=["change_pct", "eph_change_pct"]))
    # Marker area grows with the zone's pickups before the toll, so a
    # thinly used zone can't look as important as Midtown
    daily = rows["daily_before"].to_numpy()
    area = sizes[0] + (sizes[1] - sizes[0]) * (np.sqrt(daily)
                                               / np.sqrt(daily.max()))
    fig, ax = plt.subplots(figsize=(3.1, 2.8), layout="constrained")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    # The control zones are the background the cordon is read against, so
    # they are drawn first and fainter
    for group, alpha in [("control", 0.55), ("ring", 0.85), ("cbd", 0.85)]:
        shown = rows["zone_group"] == group
        ax.scatter(rows.loc[shown, "change_pct"],
                   rows.loc[shown, "eph_change_pct"],
                   s=area[shown.to_numpy()], color=ZONE_COLOURS[group],
                   alpha=alpha, linewidth=0.4, edgecolor="white",
                   label=ZONE_GROUP_NAMES[group])
    for name, offset in (annotate or {}).items():
        point = rows[rows["Zone"] == name]
        if point.empty:
            continue
        ax.annotate(name, xy=(point["change_pct"].iloc[0],
                              point["eph_change_pct"].iloc[0]),
                    xytext=offset, textcoords="offset points", fontsize=9,
                    color="#52514e",
                    ha="left" if offset[0] >= 0 else "right",
                    va="bottom" if offset[1] >= 0 else "top")
    ax.set_xlabel("Change in pickups (%)")
    ax.set_ylabel("Change in median earnings\nper engaged hour (%)")
    ax.grid(axis="both")
    handles, names = ax.get_legend_handles_labels()
    # The lower left corner holds points, so the legend gets a background
    ax.legend(handles[::-1], names[::-1], loc="lower left",
              handletextpad=0.2, borderpad=0.3, labelspacing=0.3,
              markerscale=0.7, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none")
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


def _panel_scale(limit):
    """Colour scale for one heatmap panel.

    A number gives the usual diverging scale, -limit to +limit centred at
    0. A ``(low, high)`` pair whose values have one sign gives a sequential
    scale over that range instead: with every cell on one side of 0, half a
    diverging scale is never drawn, and the panel uses colours a reader
    can barely tell apart.

    Args:
        limit (float or tuple): Scale limit, or (low, high).

    Returns:
        tuple: (norm, colormap, colourbar extend).
    """
    if not isinstance(limit, (list, tuple)):
        return (colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                DIVERGING, "both")
    low, high = limit
    colormap = SEQUENTIAL_FALL if high <= 0 else SEQUENTIAL_RISE
    return colors.Normalize(vmin=low, vmax=high), colormap, "min"


def band_day_heatmaps(tables, titles, limits, labels):
    """Heatmaps of change by time band (rows) and day of week (columns).

    Each panel gets its own colour scale when the panels are given
    different limits or labels, because a scale wide enough for the
    largest panel leaves a smaller one almost flat: the numbers are still
    printed, but the colours stop carrying the pattern.

    Args:
        tables (list of pandas.DataFrame): Outputs of
            ``explore.relative_change`` or ``explore.band_day_table``.
        titles (list of str): Panel titles.
        limits (float or list of float): Colour scale runs from -limit to
            +limit (%), one limit for every panel or one for each.
        labels (str or list of str): Colour bar label, shared or one for
            each panel.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if not isinstance(limits, (list, tuple)):
        limits = [limits] * len(tables)
    if isinstance(labels, str):
        labels = [labels] * len(tables)
    shared = (len(set(map(str, limits))) == 1
              and len(set(labels)) == 1)
    height = 2.8 if shared else 3.0
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5, height),
                             sharey=True, layout="constrained",
                             squeeze=False)
    for ax, table, title, limit, label in zip(axes[0], tables, titles,
                                              limits, labels):
        norm, colormap, extend = _panel_scale(limit)
        values = table.to_numpy(dtype=float)
        ax.imshow(values, cmap=colormap, norm=norm, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            if np.isnan(value):
                continue
            ax.text(col, row, f"{value:+.0f}", ha="center", va="center",
                    fontsize=10,
                    color=_text_colour(colormap(norm(value))))
        ax.set_xticks(range(table.shape[1]),
                      [DAY_NAMES[d - 1] for d in table.columns])
        ax.set_yticks(range(table.shape[0]),
                      [BAND_NAMES.get(b, b) for b in table.index])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)
        if not shared:
            bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
                               ax=ax, orientation="horizontal", shrink=0.9,
                               extend=extend, pad=0.02)
            bar.set_label(label)
            bar.outline.set_visible(False)
    if shared:
        norm, colormap, extend = _panel_scale(limits[0])
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
                           ax=axes[0].tolist(), extend=extend, shrink=0.9)
        bar.set_label(labels[0])
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


def monthly_lines(table, ylabel, title, toll_date):
    """One line per trip group of a monthly table.

    Args:
        table (pandas.DataFrame): Rows are months (dates), columns are
            trip groups.
        ylabel (str): y-axis label.
        title (str): Figure title.
        toll_date (datetime.date): Where to draw the toll line.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, ax = plt.subplots(layout="constrained")
    for column in table.columns:
        ax.plot(table.index, table[column], marker="o", markersize=3,
                label=GROUP_NAMES[column])
    ax.axvline(pd.Timestamp(toll_date), color="black", linestyle="--",
               linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Section 6: models
# ---------------------------------------------------------------------------
def event_study(tables, toll_date):
    """Figure 4: monthly toll effects with 95% intervals, one panel per
    service.

    Args:
        tables (dict): Service -> output of ``models.event_study``.
        toll_date (datetime.date): Where to draw the toll line.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(6.5, 3.4), sharex=True,
                             layout="constrained", squeeze=False)
    for ax, (service, table) in zip(axes[0], tables.items()):
        for group in ["treated", "ring_adjacent", "ring_across"]:
            rows = table[table["group"] == group]
            line, = ax.plot(rows["month"], rows["effect_pct"], marker="o",
                            markersize=2.5, label=GROUP_NAMES[group])
            ax.fill_between(rows["month"], rows["ci_low_pct"],
                            rows["ci_high_pct"], color=line.get_color(),
                            alpha=0.2, linewidth=0)
        ax.axvline(pd.Timestamp(toll_date), color="black", linestyle="--",
                   linewidth=0.8)
        ax.axhline(0, color="#8a8983", linewidth=0.6)
        ax.set_title(SERVICE_NAMES[service])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0][0].set_ylabel("Trips against control (%),\n"
                          "relative to December 2024")
    fig.legend(*axes[0][0].get_legend_handles_labels(),
               loc="outside lower center", ncols=3)
    return fig
