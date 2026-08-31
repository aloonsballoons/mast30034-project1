"""Label every taxi zone as ``cbd``, ``ring`` or ``control``, and finer.

- ``cbd``: one of the 38 zones on the MTA's list of zones inside the
  congestion relief zone.
- ``ring``: not in the CBD, but touching it or within ``RING_DISTANCE_FT``
  of it. The distance catches zones across the rivers, such as Long Island
  City and Downtown Brooklyn.
- ``control``: every other zone.

``zone_group_fine`` splits the ring and the control group in two:

- ``ring_adjacent``: ring zones that touch the CBD (distance 0), all in
  Manhattan.
- ``ring_across``: ring zones reached across water (distance > 0).
- ``control_near``: control zones within ``NEAR_DISTANCE_FT`` of the CBD.
- ``control_far``: every other control zone.

The labels are built before the taxi data is cleaned, because the cleaning
step uses them to give each trip a group.
"""

import geopandas as gpd
import pandas as pd

from scripts import config

# About 1 km. The shapefile is in EPSG:2263 (NY State Plane), measured in feet
RING_DISTANCE_FT = 3281
# About 2 km. Control zones this close to the CBD are ``control_near``
NEAR_DISTANCE_FT = 6562

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
        ``distance_to_cbd_ft``, ``zone_group`` and ``zone_group_fine``.
        Zones 264 and 265 (unknown and outside NYC) have no shape, so their
        groups are missing.
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

    # Distances are rounded to whole feet, but the six touching zones are
    # exactly 0 and the next nearest ring zone is 908 ft away
    distance = shapes["distance_to_cbd_ft"]
    ring, control = (shapes["zone_group"] == g for g in ["ring", "control"])
    shapes["zone_group_fine"] = shapes["zone_group"]
    shapes.loc[ring & (distance == 0), "zone_group_fine"] = "ring_adjacent"
    shapes.loc[ring & (distance > 0), "zone_group_fine"] = "ring_across"
    shapes.loc[control & (distance <= NEAR_DISTANCE_FT),
               "zone_group_fine"] = "control_near"
    shapes.loc[control & (distance > NEAR_DISTANCE_FT),
               "zone_group_fine"] = "control_far"

    labels = lookup.merge(
        shapes[["LocationID", "distance_to_cbd_ft", "zone_group",
                "zone_group_fine"]],
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
    labels = pd.read_csv(LABELS_FILE)
    # Files saved before the finer group was added are rebuilt
    if "zone_group_fine" not in labels.columns:
        return build_zone_labels()
    return labels
