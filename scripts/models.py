"""The two models in Section 6 of ``main.ipynb``.

1. **Model 1, difference-in-differences** (pyfixest). ``did_data`` builds one
   service's regression table from the model table. ``fit_did`` fits trips
   per cell with a Poisson regression, or log median earnings per engaged
   hour with weighted OLS, both with a fixed effect for each pickup zone x
   drop-off group x time band and for each date x time band, and standard
   errors clustered by pickup zone. ``did_checks`` fits the headline model
   and every check on it; ``event_study`` fits the month-by-month version.
2. **Model 2, LightGBM forecast**. ``validate`` trains on 2023 and tests on
   2024 against a 52-week baseline. ``forecast`` refits on 2023-2024 and
   predicts the toll period, ``forecast_effect`` turns the gap between
   actual and forecast into an effect comparable with Model 1, and
   ``shap_importance`` ranks the inputs.
"""

import re
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyfixest as pf
import shap
from pyspark.sql import functions as F

from scripts import config
from scripts.explore import PERIODS

# Daily Central Park weather, used by both models
WEATHER = ["rain_mm", "snow_mm", "snow_depth_mm", "temp_max_c", "temp_min_c"]
# A cell of the summary table, apart from its date. Model 1 gives each cell
# its own fixed effect, and Model 2 its own usual trip level
CELL = ["PULocationID", "dropoff_group", "time_band"]
RING_HALVES = ["ring_adjacent", "ring_across"]


def _service_rows(table, service, columns):
    """One service's rows of the model table: the cell, the date and
    ``columns``.

    The group columns become categorical, which keeps the copy small.
    """
    rows = table.loc[table["service"] == service, CELL + ["date"] + columns]
    groups = rows.columns.intersection(
        ["dropoff_group", "trip_group", "trip_group_coarse"])
    return rows.astype(dict.fromkeys(groups, "category")).reset_index(
        drop=True)


# ---------------------------------------------------------------------------
# 1. Model 1: difference-in-differences
# ---------------------------------------------------------------------------
# Toll terms of each model. fit_did adds treated x weather and the fixed
# effects
HEADLINE = "treated_after + ring_adjacent_after + ring_across_after"
POOLED_RING = "treated_after + ring_after"
BY_BAND = "i(time_band, treated_after) + i(time_band, ring_after)"
TRENDS = (HEADLINE
          + " + treated:trend + ring_adjacent:trend + ring_across:trend")
# Standard errors clustered by pickup zone, because days from the same zone
# are not independent
VCOV = {"CRV1": "PULocationID"}


def set_toll_date(data, toll_date):
    """Add the after-toll terms for a given toll date.

    Args:
        data (pandas.DataFrame): Output of ``did_data``, or rows of it.
        toll_date (datetime.date): The real or a placebo toll date.

    Returns:
        pandas.DataFrame: ``data`` with ``treated_after``, ``ring_after``,
        ``ring_adjacent_after`` and ``ring_across_after`` (1 for that
        group's rows from ``toll_date`` on, otherwise 0).
    """
    after = (data["date"] >= pd.Timestamp(toll_date)).astype(float)
    return data.assign(**{f"{group}_after": data[group] * after
                          for group in ["treated", "ring"] + RING_HALVES})


def did_data(table, service):
    """One service's rows of the model table, ready for ``fit_did``.

    Args:
        table (pandas.DataFrame): The model table (Section 4).
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pandas.DataFrame: The rows with the fixed effect ids ``cell`` and
        ``date_band``, 0/1 columns for each trip group (``ring`` is both
        halves), ``trend`` (years since the first date), ``log_eph`` (log
        of the median earnings per engaged hour, missing when the cell has
        none) and the after-toll terms of ``set_toll_date``.
    """
    data = _service_rows(table, service, [
        "trip_group", "trips", "trips_with_earnings",
        "median_earnings_per_engaged_hour"] + WEATHER)
    data = data.assign(
        cell=data.groupby(CELL, observed=True).ngroup(),
        date_band=data.groupby(["date", "time_band"], observed=True).ngroup(),
        trend=(data["date"] - data["date"].min()).dt.days / 365.25,
        log_eph=np.log(data["median_earnings_per_engaged_hour"]),
        **{group: (data["trip_group"] == group).astype(float)
           for group in ["treated"] + RING_HALVES})
    data["ring"] = data["ring_adjacent"] + data["ring_across"]
    return set_toll_date(data, config.TOLL_START_DATE)


def fit_did(data, terms, outcome="trips"):
    """Fit one difference-in-differences model.

    Every model also has ``treated`` x each weather column, so a weather
    effect that differs inside the CBD isn't read as a toll effect, and a
    fixed effect for each ``cell`` and each ``date_band``. The date fixed
    effects absorb the after-toll flag, day of week, holidays and citywide
    weather, separately for each time band.

    Args:
        data (pandas.DataFrame): Output of ``did_data``, or rows of it.
        terms (str): The toll terms, such as ``HEADLINE``.
        outcome (str): ``"trips"`` for a Poisson regression of trips, or
            ``"log_eph"`` for OLS on log median earnings per engaged hour,
            weighted by the trips with known earnings so that every trip
            counts once.

    Returns:
        pyfixest model: The fitted model.
    """
    weather = " + ".join(f"treated:{column}" for column in WEATHER)
    formula = f"{outcome} ~ {terms} + {weather} | cell + date_band"
    with warnings.catch_warnings():
        # pyfixest drops rows that can't inform the estimates and warns:
        # cells with no trips on any day (Poisson "separation") and cells
        # with a single row ("singleton"). ``model._N`` counts the rows kept
        warnings.filterwarnings("ignore", message=".*(separation|singleton)")
        if outcome == "trips":
            return pf.fepois(formula, data=data, vcov=VCOV)
        return pf.feols(formula, data=data.dropna(subset=[outcome]),
                        weights="trips_with_earnings", vcov=VCOV)


def _pct(log_change):
    """A change on the log scale as a percentage, 100 x (e^b - 1)."""
    return (np.exp(log_change) - 1) * 100


def _split_term(name):
    """Split a coefficient name into its trip group and term.

    ``treated_after`` -> (treated, all), ``time_band::midday:ring_after``
    -> (ring, midday), ``treated:trend`` -> (treated, trend) and
    ``treated:rain_mm`` -> (treated, rain_mm).
    """
    match = re.fullmatch(r"(?:time_band::(\w+):)?(\w+)_after", name)
    if match:
        return match.group(2), match.group(1) or "all"
    group, variable = name.split(":")
    return group, variable


def effects(model):
    """The toll (and trend) coefficients of a model, as % changes.

    Args:
        model: A model from ``fit_did``.

    Returns:
        pandas.DataFrame: Indexed by ``group`` and ``term`` (``all`` for a
        whole-day effect, a time band, or ``trend`` for the change per
        year), with ``effect_pct``, ``ci_low_pct`` and ``ci_high_pct``
        (95%). The weather terms are left out.
    """
    tidy = model.tidy()
    table = pd.DataFrame({
        "effect_pct": _pct(tidy["Estimate"]),
        "ci_low_pct": _pct(tidy["2.5%"]),
        "ci_high_pct": _pct(tidy["97.5%"])})
    table.index = pd.MultiIndex.from_tuples(
        [_split_term(name) for name in tidy.index], names=["group", "term"])
    return table[~table.index.get_level_values("term").isin(WEATHER)]


def did_checks(data):
    """Fit the headline model, its variants and the checks on it.

    Each fit is run for trips and for earnings per engaged hour:

    - ``headline``: the toll effect on treated trips and on each ring half
    - ``pooled ring``: one ring term instead of two
    - ``by time band``: one treated and one (pooled) ring term per band
    - ``placebo``: 2023-2024 only, with a fake toll on 5 January 2024
    - ``2024-2025 only``: the headline model without 2023
    - ``linear trends``: the headline model plus a linear time trend for
      the treated group and each ring half, so the toll terms measure the
      break from each group's trend
    - ``buffer band dropped``: the headline model without ``control_near``
      trips, so control trips have both ends more than 2 km from the CBD

    Args:
        data (pandas.DataFrame): Output of ``did_data``.

    Returns:
        pandas.DataFrame: The outputs of ``effects`` for every fit, with
        ``outcome`` and ``check`` index levels in front and ``rows``, the
        rows each fit used.
    """
    year = data["date"].dt.year
    # Rows to keep (None = all) and toll terms of each fit
    fits = {
        "headline": (None, HEADLINE),
        "pooled ring": (None, POOLED_RING),
        "by time band": (None, BY_BAND),
        "placebo": (year < config.TOLL_START_DATE.year, HEADLINE),
        "2024-2025 only": (year >= config.PLACEBO_DATE.year, HEADLINE),
        "linear trends": (None, TRENDS),
        "buffer band dropped": (data["trip_group"] != "control_near",
                                HEADLINE),
    }
    tables = {}
    for outcome in ["trips", "log_eph"]:
        for check, (keep, terms) in fits.items():
            rows = data if keep is None else data[keep]
            if check == "placebo":
                rows = set_toll_date(rows, config.PLACEBO_DATE)
            model = fit_did(rows, terms, outcome)
            tables[outcome, check] = effects(model).assign(rows=model._N)
    return pd.concat(tables, names=["outcome", "check"])


def event_study(table, service):
    """Month-by-month toll effects on trips (Figure 4).

    One Poisson regression with a treated x month, ring_adjacent x month
    and ring_across x month term for every month but the last one before
    the toll (December 2024 in the full run), which is the reference. The
    trips are summed to months first: 105 month terms on 5.8M daily rows
    would need about 5 GB per copy of the data. Fixed effects are each
    cell and each month x time band. There is no weather term, because
    monthly weather would be one value per month, which the month terms
    already absorb.

    Args:
        table (pandas.DataFrame): The model table.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pandas.DataFrame: ``group``, ``month`` (first day), ``effect_pct``,
        ``ci_low_pct`` and ``ci_high_pct`` (all 0 in the reference month).
    """
    rows = _service_rows(table, service, ["trip_group", "trips"])
    monthly = (rows.assign(month=rows["date"].dt.strftime("%Y-%m"))
               .groupby(CELL + ["trip_group", "month"], observed=True)
               ["trips"].sum().reset_index())
    groups = ["treated"] + RING_HALVES
    monthly = monthly.assign(
        cell=monthly.groupby(CELL, observed=True).ngroup(),
        month_band=monthly.groupby(["month", "time_band"],
                                   observed=True).ngroup(),
        **{group: (monthly["trip_group"] == group).astype(float)
           for group in groups})
    toll_month = config.TOLL_START_DATE.strftime("%Y-%m")
    base = monthly.loc[monthly["month"] < toll_month, "month"].max()
    terms = " + ".join(f"i(month, {group}, ref='{base}')" for group in groups)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(separation|singleton)")
        model = pf.fepois(f"trips ~ {terms} | cell + month_band",
                          data=monthly, vcov=VCOV)

    tidy = model.tidy()
    names = tidy.index.str.extract(r"month::([\d-]+):(\w+)")
    result = pd.DataFrame({
        "group": names[1].to_numpy(), "month": names[0].to_numpy(),
        "effect_pct": _pct(tidy["Estimate"]).to_numpy(),
        "ci_low_pct": _pct(tidy["2.5%"]).to_numpy(),
        "ci_high_pct": _pct(tidy["97.5%"]).to_numpy()})
    reference = pd.DataFrame({"group": groups, "month": base,
                              "effect_pct": 0.0, "ci_low_pct": 0.0,
                              "ci_high_pct": 0.0})
    result = pd.concat([result, reference], ignore_index=True)
    result["month"] = pd.to_datetime(result["month"])
    return result.sort_values(["group", "month"]).reset_index(drop=True)


def median_eph_before(trips):
    """Median earnings per engaged hour of treated trips before the toll.

    The median is over every treated trip in the 2024 period (5 January to
    31 December 2024, as in Section 5), with ``percentile_approx`` at
    accuracy 1,000. It turns Model 1's percentage into dollars.

    Args:
        trips (pyspark.sql.DataFrame): Output of ``clean.add_columns``.

    Returns:
        float: The median, in dollars per engaged hour.
    """
    start, end = PERIODS["before"]
    treated = ((F.col("trip_group") == "treated")
               & F.col("date").between(F.lit(start), F.lit(end)))
    return (trips.where(treated)
            .agg(F.percentile_approx("earnings_per_engaged_hour", 0.5, 1000))
            .first()[0])


# ---------------------------------------------------------------------------
# 2. Model 2: LightGBM forecast
# ---------------------------------------------------------------------------
# Trips are counts, so a Poisson objective. Total earnings are 0 in cells
# with no trips and continuous otherwise, which a Tweedie objective fits
OBJECTIVES = {"trips": "poisson", "total_earnings": "tweedie"}
LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 63,
               "random_state": 0, "verbose": -1}
# Columns of the model table that ``features`` needs besides the cell and date
INPUTS = ["day_of_week", "is_holiday"] + WEATHER


def features(data, train, target):
    """Model 2's inputs for every row of one service.

    Zone, drop-off group, time band, day of week and month are categorical.
    Holiday, weather and ``year`` are numeric. ``usual`` is the cell's
    mean ``target`` over the training rows only, so no test or toll-period
    data leaks in. ``year`` lets the forecast follow the change from 2023
    to 2024 (Section 6.7); trees can't extrapolate, so 2025 is forecast as
    a year like 2024.

    Args:
        data (pandas.DataFrame): One service's rows of the model table.
        train (pandas.Series): Boolean mask of the training rows.
        target (str): ``"trips"`` or ``"total_earnings"``.

    Returns:
        pandas.DataFrame: One row of inputs per row of ``data``.
    """
    usual = (data[train].groupby(CELL, observed=True)[target].mean()
             .rename("usual"))
    return pd.DataFrame({
        "PULocationID": data["PULocationID"].astype("category"),
        "dropoff_group": data["dropoff_group"],
        "time_band": data["time_band"],
        "day_of_week": data["day_of_week"].astype("category"),
        "month": data["date"].dt.month.astype("category"),
        "is_holiday": data["is_holiday"].astype(int),
        **{column: data[column] for column in WEATHER},
        "year": data["date"].dt.year,
        "usual": data.join(usual, on=CELL)["usual"],
    })


def _fit(inputs, target_values, target):
    """Fit one LightGBM model with ``LGBM_PARAMS``."""
    model = lgb.LGBMRegressor(objective=OBJECTIVES[target], **LGBM_PARAMS)
    return model.fit(inputs, target_values)


def _accuracy(actual, predicted):
    """MAE, RMSE and WMAPE (% of all actual values) of a forecast."""
    error = predicted - actual
    return {"mae": error.abs().mean(), "rmse": np.sqrt((error ** 2).mean()),
            "wmape_pct": error.abs().sum() / actual.sum() * 100}


def validate(table, service):
    """Train on the first year, test on the second, against a baseline.

    In the full run that is 2023 and 2024, both before the toll, so the
    test measures how well the model forecasts a year it hasn't seen. The
    baseline is the same cell (zone, drop-off group, time band) on the same
    weekday 52 weeks earlier.

    Args:
        table (pandas.DataFrame): The model table.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) - MAE, RMSE and WMAPE
        by target and forecast (model or baseline), and the test rows'
        ``PULocationID``, ``date``, ``trips`` and ``predicted`` trips.
    """
    data = _service_rows(table, service, INPUTS + list(OBJECTIVES))
    first, second = sorted(data["date"].dt.year.unique())[:2]
    train = data["date"].dt.year == first
    test = data["date"].dt.year == second
    scores = {}
    for target in OBJECTIVES:
        inputs = features(data, train, target)
        model = _fit(inputs[train], data.loc[train, target], target)
        predicted = pd.Series(model.predict(inputs[test]),
                              index=data.index[test])
        earlier = (data[CELL + ["date", target]]
                   .assign(date=data["date"] + pd.Timedelta(weeks=52)))
        baseline = (data.loc[test, CELL + ["date"]]
                    .merge(earlier, on=CELL + ["date"], how="left")[target]
                    .set_axis(predicted.index))
        actual = data.loc[test, target]
        scores[target, "model"] = _accuracy(actual, predicted)
        scores[target, "52-week baseline"] = _accuracy(actual, baseline)
        if target == "trips":
            rows = data.loc[test, ["PULocationID", "date", "trips"]].assign(
                predicted=predicted)
    scores = pd.DataFrame(scores).T.rename_axis(["target", "forecast"])
    return scores, rows


def zone_errors(table, rows, service, n=8):
    """The pickup zones with the largest test errors in trips.

    Args:
        table (pandas.DataFrame): The model table.
        rows (pandas.DataFrame): Second output of ``validate``.
        service (str): The service ``rows`` came from.
        n (int): Number of zones.

    Returns:
        pandas.DataFrame: For the ``n`` zones with the largest sum of
        absolute errors: ``abs_error_share_pct`` (share of all absolute
        errors), ``trips`` and ``predicted`` in the test year,
        ``bias_pct`` (predicted against actual) and ``change_pct`` (actual
        trips in the test year against the training year).
    """
    rows = rows.assign(abs_error=(rows["predicted"] - rows["trips"]).abs())
    zones = rows.groupby("PULocationID")[["trips", "predicted",
                                          "abs_error"]].sum()
    zones["abs_error_share_pct"] = (zones["abs_error"]
                                    / zones["abs_error"].sum() * 100)
    zones["bias_pct"] = (zones["predicted"] / zones["trips"] - 1) * 100
    test_year = rows["date"].dt.year.iloc[0]
    data = table[(table["service"] == service)
                 & (table["date"].dt.year == test_year - 1)]
    before = data.groupby("PULocationID")["trips"].sum()
    zones["change_pct"] = (zones["trips"] / before - 1) * 100
    return (zones.nlargest(n, "abs_error")
            [["abs_error_share_pct", "trips", "predicted", "bias_pct",
              "change_pct"]])


def forecast(table, service, target):
    """Refit on every year before the toll and forecast the toll period.

    Args:
        table (pandas.DataFrame): The model table.
        service (str): ``"yellow"`` or ``"fhvhv"``.
        target (str): ``"trips"`` or ``"total_earnings"``.

    Returns:
        tuple: (lightgbm.LGBMRegressor, pandas.DataFrame, pandas.DataFrame)
        - the model, its inputs for the toll-period rows, and those rows'
        ``trip_group``, ``trip_group_coarse``, ``time_band``, ``actual``
        and ``predicted`` values.
    """
    data = _service_rows(table, service, INPUTS + [
        "trip_group", "trip_group_coarse", target])
    train = data["date"].dt.year < config.TOLL_START_DATE.year
    after = data["date"] >= pd.Timestamp(config.TOLL_START_DATE)
    inputs = features(data, train, target)
    model = _fit(inputs[train], data.loc[train, target], target)
    rows = data.loc[after, ["trip_group", "trip_group_coarse",
                            "time_band"]].assign(
        actual=data.loc[after, target],
        predicted=model.predict(inputs[after]))
    return model, inputs[after], rows


def forecast_effect(rows, group="trip_group", by=()):
    """Actual against forecast, and the same relative to control trips.

    The gap for control trips includes everything that changed citywide in
    2025, such as rideshare growth, so the toll effect is the group's gap
    relative to control: (1 + group gap) / (1 + control gap) - 1. That is
    the same scale as Model 1's 100 x (e^b - 1).

    Args:
        rows (pandas.DataFrame): Third output of ``forecast``.
        group (str): ``"trip_group"`` or ``"trip_group_coarse"``.
        by (tuple of str): Columns to split by as well, such as
            ``("time_band",)``.

    Returns:
        pandas.DataFrame: ``gap_pct`` (actual against forecast) and
        ``effect_pct`` (relative to all control trips) by ``by`` and
        ``group``.
    """
    by = list(by)

    def ratio(part, keys):
        sums = part.groupby(keys, observed=True)[["actual", "predicted"]].sum()
        return sums["actual"] / sums["predicted"]

    table = ratio(rows, by + [group]).to_frame("ratio")
    control = rows[rows["trip_group_coarse"] == "control"]
    if by:
        control_ratio = ratio(control, by).reindex(
            table.index.droplevel(group)).to_numpy()
    else:
        control_ratio = control["actual"].sum() / control["predicted"].sum()
    table["gap_pct"] = (table["ratio"] - 1) * 100
    table["effect_pct"] = (table["ratio"] / control_ratio - 1) * 100
    return table[["gap_pct", "effect_pct"]]


def shap_importance(model, inputs, n=20_000):
    """Mean absolute SHAP value of each input, on a random sample of rows.

    SHAP values split each forecast into one part per input (Lundberg and
    Lee 2017). For a Poisson or Tweedie model they are on the log scale.
    TreeSHAP's run time grows with the number of rows, so it runs on a
    sample rather than all 1.9M toll-period rows of a service.

    Args:
        model (lightgbm.LGBMRegressor): A model from ``forecast``.
        inputs (pandas.DataFrame): Its toll-period inputs.
        n (int): Rows to sample (fixed seed).

    Returns:
        pandas.Series: Mean |SHAP| by input.
    """
    sample = inputs.sample(min(n, len(inputs)), random_state=0)
    values = shap.TreeExplainer(model).shap_values(sample)
    return pd.Series(np.abs(values).mean(axis=0), index=inputs.columns)


# ---------------------------------------------------------------------------
# 3. The two models side by side (Table 2)
# ---------------------------------------------------------------------------
# Table 2's rows: the treated effect for the whole day and each time band,
# then the effect on each ring half
TABLE_2_ROWS = ([("treated", "all")]
                + [("treated", band) for band in
                   ["overnight", "morning_peak", "midday", "evening"]]
                + [(half, "all") for half in RING_HALVES])


def _model_1(effects, outcome):
    """Model 1's Table 2 rows: headline fit, then the time-band fit."""
    fits = effects.loc[outcome]
    return (pd.concat([fits.loc["headline"], fits.loc["by time band"]])
            .reindex(TABLE_2_ROWS)[["effect_pct", "ci_low_pct",
                                    "ci_high_pct"]])


def _model_2(rows):
    """Model 2's Table 2 rows: by trip group, then treated by time band."""
    whole = forecast_effect(rows).assign(term="all").set_index(
        "term", append=True)
    bands = forecast_effect(rows, "trip_group_coarse", ["time_band"])
    bands = bands.swaplevel().rename_axis(whole.index.names)
    return (pd.concat([whole, bands]).rename_axis(["group", "term"])
            .reindex(TABLE_2_ROWS)["effect_pct"])


def comparison(effects, forecasts):
    """Table 2 for one service: both models' toll effects side by side.

    Args:
        effects (pandas.DataFrame): One service's output of ``did_checks``.
        forecasts (dict): Target -> third output of ``forecast`` for the
            same service.

    Returns:
        pandas.DataFrame: One row per ``TABLE_2_ROWS`` entry. Model 1's
        effects on trips and on earnings per engaged hour with 95%
        intervals, and Model 2's effects on trips and total earnings, all
        in % relative to control trips.
    """
    trips, eph = _model_1(effects, "trips"), _model_1(effects, "log_eph")
    return pd.DataFrame({
        "Trips, Model 1 (%)": trips["effect_pct"],
        "Trips, Model 1, 95% CI low": trips["ci_low_pct"],
        "Trips, Model 1, 95% CI high": trips["ci_high_pct"],
        "Trips, Model 2 (%)": _model_2(forecasts["trips"]),
        "Earnings per engaged hour, Model 1 (%)": eph["effect_pct"],
        "Earnings per engaged hour, Model 1, 95% CI low": eph["ci_low_pct"],
        "Earnings per engaged hour, Model 1, 95% CI high": eph["ci_high_pct"],
        "Total earnings, Model 2 (%)": _model_2(forecasts["total_earnings"]),
    })
