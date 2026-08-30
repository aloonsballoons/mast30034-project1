"""Label every taxi zone as ``cbd``, ``ring`` or ``control``.

- ``cbd``: one of the 38 zones on the MTA's list of zones inside the
  congestion relief zone.
- ``ring``: not in the CBD, but touching it or within ``RING_DISTANCE_FT``
  of it. The distance catches zones across the rivers, such as Long Island
  City and Downtown Brooklyn.
- ``control``: every other zone.

The labels are built before the taxi data is cleaned, because the cleaning
step uses them to give each trip a group.
"""

import geopandas as gpd
import pandas as pd

from scripts import config

# About 1 km. The shapefile is in EPSG:2263 (NY State Plane), measured in feet
RING_DISTANCE_FT = 3281

LABELS_FILE = config.CURATED_DIR / "zone_labels.csv"


def load_zone_shapes():
    """Load the taxi zone shapefile in NY State Plane (feet).

    Returns:
        geopandas.GeoDataFrame: One row per zone (1-263) with
        ``LocationID``, ``zone``, ``borough`` and ``geometry``.
    """
    shapes = gpd.read_file(config.RAW_DIR / "taxi_zones" / "taxi_zones.shp")
    shapes = shapes.to_crs(epsg=2263)
    # Check that no zone ID has more than one shape
    assert shapes["LocationID"].is_unique, "a zone ID appears twice"
    return shapes[["LocationID", "zone", "borough", "geometry"]]


def shared_names():
    """Find zone IDs that share a name or a shape in the shapefile.

    Several taxi zones carry the same name: 56 and 57 are both "Corona",
    and 103, 104 and 105 are all "Governor's Island/Ellis Island/Liberty
    Island". In this version of the shapefile they are separate polygons,
    checked here, so the only risk is grouping by name instead of
    ``LocationID``, which would merge them on a map or in a join. Every
    join in the project uses ``LocationID``.

    Returns:
        pandas.DataFrame: One row per zone whose name is used by more than
        one ``LocationID``, with ``LocationID``, ``zone``, ``borough`` and
        ``area_sq_mi``.
    """
    shapes = load_zone_shapes()
    # 27,878,400 square feet in a square mile
    shapes["area_sq_mi"] = (shapes.area / 27_878_400).round(3)
    assert not shapes.geometry.to_wkb().duplicated().any(), \
        "two zone IDs have an identical shape"
    repeated = shapes["zone"].duplicated(keep=False)
    return (shapes.loc[repeated, ["LocationID", "zone", "borough",
                                  "area_sq_mi"]]
            .sort_values(["zone", "LocationID"]).reset_index(drop=True))


def build_zone_labels():
    """Label every zone and save the table to ``data/curated/``.

    Returns:
        pandas.DataFrame: One row per zone in the lookup table (1-265) with
        ``LocationID``, ``Borough``, ``Zone``, ``in_cbd``,
        ``distance_to_cbd_ft`` and ``zone_group``. Zones 264 and 265
        (unknown and outside NYC) have no shape, so their group is missing.
    """
    lookup = pd.read_csv(config.RAW_DIR / "taxi_zones"
                         / "taxi_zone_lookup.csv")
    cbd_ids = set(pd.read_csv(config.RAW_DIR / "mta"
                              / "cbd_taxi_zones.csv")["taxi_zone"])

    shapes = load_zone_shapes()
    in_cbd = shapes["LocationID"].isin(cbd_ids)
    cbd_shape = shapes.loc[in_cbd, "geometry"].union_all()
    shapes["distance_to_cbd_ft"] = shapes.distance(cbd_shape).round(0)

    shapes["zone_group"] = "control"
    shapes.loc[shapes["distance_to_cbd_ft"] <= RING_DISTANCE_FT,
               "zone_group"] = "ring"
    shapes.loc[in_cbd, "zone_group"] = "cbd"

    labels = lookup.merge(
        shapes[["LocationID", "distance_to_cbd_ft", "zone_group"]],
        on="LocationID", how="left")
    labels.insert(3, "in_cbd", labels["LocationID"].isin(cbd_ids))
    assert labels["in_cbd"].sum() == len(cbd_ids) == 38
    labels.to_csv(LABELS_FILE, index=False)
    return labels


def load_zone_labels():
    """Load the saved zone labels, building them first if needed.

    Returns:
        pandas.DataFrame: The output of ``build_zone_labels``.
    """
    if not LABELS_FILE.exists():
        return build_zone_labels()
    return pd.read_csv(LABELS_FILE)
