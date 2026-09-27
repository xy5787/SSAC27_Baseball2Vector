"""A Marcel-style wRC+ projector, following Tom Tango's original definition.

Sources the parameters below were taken from, rather than from memory:

* Tom Tango, "Marcel the Monkey Forecasting System: Batting Projections"
  https://www.tangotiger.net/archives/stud0346.shtml
      - "Weight each season as 5/4/3. 2003 counts as '5' and 2001 counts as '3'."
      - "I then forced in that player's league average to come in at a total of
        1200 PA for each player (2 weights x 600 PA)."
      - "Determine the projected PA = 0.5 * 2004PA + 0.1 * 2003PA + 200."
      - "If over 29, AgeAdj = (age - 29) * .003. If under 29,
        AgeAdj = (age - 29) * .006"
      - "Any 2004 rookie with no MLB experience will project at the league
        average."
* Baseball-Reference's description of the same system
  https://www.baseball-reference.com/about/marcels.shtml
* bdilday/marcelR, the widely used reference implementation, consulted to
  disambiguate two things Tango's prose leaves open -- the scale the 1200
  constant lives on, and the sign convention of the age term:
  https://github.com/bdilday/marcelR  (`R/marcelBatting.R`, `R/utilities.R`)

Adopted parameters, and why each one is what it is:

1. **Season weights 5 / 4 / 3** on seasons t, t-1, t-2 when projecting t+1.
   Tango's own numbers, used raw (not renormalized to sum to 1). marcelR keeps
   them raw: `x_pa <- x_pa + pa * metric_weights[idx]`.

2. **Regression constant 1200 = 100 x sum(weights)**, on the *raw-weighted* PA
   scale. marcelR: `num = x_av*100*sw + x_metric`, `denom = x_pa + 100*sw`, with
   `sw = 5+4+3 = 12`. A 600-PA-per-season regular therefore has weighted PA
   12 x 600 = 7200 and reliability 7200 / 8400 = 0.857. This is the reading that
   makes Tango's "2 weights x 600 PA" arithmetic come out at 1200.

3. **League baseline.** Two variants, selected by `league_baseline`:

   * ``"definition"`` (default, primary). The baseline is **100.0 in every
     season**, because that is what wRC+ *is*: a league- and park-adjusted index
     centred on 100. Empirically confirmed on the full batting population in the
     five designated-hitter seasons, where no pitcher plate appearances distort
     it: 2020 100.01, 2022 100.07, 2023 100.18, 2024 100.18, 2025 99.81. This
     variant needs no population definition at all, so it cannot smuggle in
     information from a season later than the one being projected.

   * ``"cohort"`` (robustness). The recency- and PA-weighted mean over the
     qualified (PA >= 100) cohort, per marcelR's
     `x_av_num += lgAv_i * w_i * (pa_i + pebble)`. This is what the first
     iteration of this package used. It is retained for comparability but is
     **not** the primary: the PA >= 100 qualifier is a selection rule, and in a
     60-game season it selects a different kind of player, which shows up as a
     2020 cohort mean of 106.38 against 102.0-102.2 in every full season.

   A third candidate -- restricting the population to players who ever reached
   100 PA within 2019-2025 -- gives the most stable series of all (100.7 to
   101.6 across seven seasons) but defines the population using seasons later
   than the origin. That is a forward reference, so it was rejected regardless
   of how well behaved it looked. See decisions_log.md D12.

4. **League re-centering.** marcelR finishes by rescaling every projection in an
   origin year so the PA-weighted mean projected rate equals the league rate.
   That step presumes the projected set *is* the league. Our focal players are a
   selected (PA >= 100) subset that is genuinely above average, so re-centering
   them onto the league mean would inject a systematic downward bias. It is
   therefore **off by default** and available via `recenter=True` only for the
   ``"cohort"`` variant, where the population and the baseline agree.

5. **Age adjustment**, applied multiplicatively to the projected *rate* using
   the age the player will be in the projected season (marcelR passes
   `data$Age + 1`):

       age > 29:  1 / (1 + 0.003 * (age - 29))     -- decline
       age < 29:  1 + 0.006 * (29 - age)           -- growth
       age = 29:  1

   Tango states the two slopes (.003 above 29, .006 below) but not the sign
   convention; marcelR's `age_adjustment()` supplies it, and it is the only
   reading under which young players are projected upward.

   The age fed in is FanGraphs' own `Age` column, i.e. conventional baseball age
   as of mid-season -- the definition Tango's constants were fitted against. Note
   this differs by up to a year from the archived `age_t = Season - birth_year`
   used as a *model feature* elsewhere in this package (they agree on 47% of
   rows); see decisions_log.md D13.

6. **Projected PA = 200 + 0.5 * PA_t + 0.1 * PA_{t-1}**. Computed and returned
   for completeness. It does not enter the rate projection, and is *not* used as
   a model feature anywhere in this package -- PA_t already appears in its own
   right in the M3/M4/M5 context models.

7. **Fewer than three prior seasons.** Missing seasons contribute zero to both
   the weighted metric and the weighted PA, so the 1200-PA regression does more
   of the work. A player with no history at all projects at the league baseline,
   which is Tango's rookie rule.

Adaptation to wRC+: it is already a league- and park-adjusted rate, so the
"counting stat" Marcel weights is taken to be `wRC+ x PA` and the projection is
the ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON_WEIGHTS: tuple[int, int, int] = (5, 4, 3)
REGRESSION_PA_PER_WEIGHT = 100  # x sum(weights) = 1200 PA of league average
PLAYING_TIME_WEIGHTS: tuple[float, float, float] = (0.5, 0.1, 0.0)
PLAYING_TIME_INTERCEPT = 200
AGE_PIVOT = 29
AGE_SLOPE_OVER = 0.003
AGE_SLOPE_UNDER = 0.006
PEBBLE = 1e-6

#: wRC+ is defined so that the league average is exactly this.
WRC_PLUS_LEAGUE_AVERAGE = 100.0

#: WAR is reported per this many plate appearances when used as a rate.
WAR_RATE_DENOMINATOR = 600

HISTORY_COLUMNS = ["player_id", "Season", "PA", "wRC+", "Age"]

#: Per-metric configuration. Marcel always projects a *rate*; ``scale`` turns
#: that rate back into the quantity being predicted.
#:
#: * ``wRC+``   already a rate indexed to 100; scale 1.
#: * ``WAR``    a counting stat. Marcel projects WAR per PA and multiplies by the
#:              projected playing time, which is exactly Tango's
#:              ``proj_value = proj_pa * proj_rate``. This is the one place the
#:              projected-PA formula does real work.
#: * ``WAR_per_600`` the same rate, expressed per 600 PA instead of multiplied
#:              back out, so tool quality is separated from playing time.
#: ``source_is_rate`` says how the stored column relates to plate appearances.
#: wRC+ is already per-PA, so its weighted numerator is ``w * PA * value`` --
#: exactly marcelR's treatment of a rate. WAR is a season total, so its
#: numerator is ``w * value`` and dividing by weighted PA yields WAR per PA.
#: Getting this wrong silently multiplies WAR by PA twice.
METRICS = {
    "wRC+": {
        "column": "wRC+",
        "source_is_rate": True,
        "rate_scale": 1.0,
        "multiply_by_projected_pa": False,
    },
    "WAR": {
        "column": "WAR",
        "source_is_rate": False,
        "rate_scale": 1.0,
        "multiply_by_projected_pa": True,
    },
    "WAR_per_600": {
        "column": "WAR",
        "source_is_rate": False,
        "rate_scale": float(WAR_RATE_DENOMINATOR),
        "multiply_by_projected_pa": False,
    },
}


def age_adjustment(age: float) -> float:
    """marcelR `age_adjustment()`: multiplicative, applied at the projected age."""
    if not np.isfinite(age) or age <= 0:
        return 1.0
    if age > AGE_PIVOT:
        return 1.0 / (1.0 + AGE_SLOPE_OVER * (age - AGE_PIVOT))
    if age < AGE_PIVOT:
        return 1.0 + AGE_SLOPE_UNDER * (AGE_PIVOT - age)
    return 1.0


def league_rates(
    history: pd.DataFrame, variant: str, metric: str = "wRC+"
) -> dict[int, float]:
    """Per-season league baseline (per PA) the projection regresses toward.

    For wRC+ the ``"definition"`` variant is the constant 100, because that is
    what the index means. WAR has no such definitional centre, so its baseline is
    always the empirical league rate -- which is stable enough to trust: league
    WAR per 600 PA over the whole batting population runs 1.833 to 1.900 across
    2019-2025, a 3.5% spread, and is unaffected by the pitcher plate appearances
    that distort a wRC+ population mean in 2019 and 2021.
    """
    column = METRICS[metric]["column"]
    seasons = sorted(int(season) for season in history["Season"].unique())
    if metric == "wRC+" and variant == "definition":
        return {season: WRC_PLUS_LEAGUE_AVERAGE for season in seasons}
    population = history[history["PA"] >= 100] if variant == "cohort" else history
    if METRICS[metric]["source_is_rate"]:
        return {
            int(season): float(np.average(block[column], weights=block["PA"]))
            for season, block in population.groupby("Season")
        }
    # Counting stat: the league baseline is the per-PA rate.
    return {
        int(season): float(block[column].sum() / block["PA"].sum())
        for season, block in population.groupby("Season")
    }


def project(
    history: pd.DataFrame,
    league_baseline: str = "definition",
    recenter: bool = False,
    metric: str = "wRC+",
) -> pd.DataFrame:
    """Marcel wRC+ projection for every (player, origin season) in *history*.

    Args:
        history: one row per player-season, columns ``player_id``, ``Season``,
            ``PA``, ``wRC+``, ``Age``. Every season a player actually played
            should be present, including seasons under the 100-PA qualifier --
            Marcel treats an absent season as zero PA, which is correct for
            "did not play" and wrong for "played a little".
        league_baseline: ``"definition"`` (100.0 every season) or ``"cohort"``.
        recenter: apply marcelR's per-origin-year rescale. Only meaningful when
            *history* is the whole population being projected.

    Returns:
        One row per (``player_id``, ``origin_season``) carrying the projection
        for ``origin_season + 1``.
    """
    config = METRICS[metric]
    metric_column = config["column"]
    missing = [
        column
        for column in [*HISTORY_COLUMNS, metric_column]
        if column not in history
    ]
    assert not missing, f"history is missing {missing}"
    assert not history.duplicated(["player_id", "Season"]).any()

    rates = league_rates(history, league_baseline, metric)
    available = set(rates)

    record: dict[tuple[object, int], tuple[float, float]] = {}
    for player_id, season, pa, rate in history[
        ["player_id", "Season", "PA", metric_column]
    ].to_numpy(dtype=object):
        record[(player_id, int(season))] = (float(pa), float(rate))

    weight_sum = sum(SEASON_WEIGHTS)
    regression_pa = REGRESSION_PA_PER_WEIGHT * weight_sum

    rows: list[dict] = []
    for player_id, origin_season, age in history[
        ["player_id", "Season", "Age"]
    ].to_numpy(dtype=object):
        origin_season = int(origin_season)
        weighted_metric = 0.0
        weighted_pa = 0.0
        league_num = 0.0
        league_denom = 0.0
        target_num = 0.0
        target_denom = 0.0
        projected_pa = float(PLAYING_TIME_INTERCEPT)
        seasons_used = 0

        for index, weight in enumerate(SEASON_WEIGHTS):
            season = origin_season - index
            if season not in available:
                # Outside the data window: contribute nothing at all, rather than
                # injecting a zero league average that would drag the baseline down.
                continue
            league_rate = rates[season]
            pa, rate = record.get((player_id, season), (0.0, 0.0))
            if pa > 0:
                seasons_used += 1
            weighted_metric += weight * (pa * rate if config["source_is_rate"] else rate)
            weighted_pa += weight * pa
            league_num += league_rate * weight * (pa + PEBBLE)
            league_denom += weight * (pa + PEBBLE)
            target_num += weight * league_rate
            target_denom += weight
            projected_pa += PLAYING_TIME_WEIGHTS[index] * pa

        league_value = league_num / league_denom
        raw_rate = (
            league_value * regression_pa + weighted_metric
        ) / (weighted_pa + regression_pa)
        adjusted_rate = age_adjustment(float(age) + 1.0) * raw_rate
        projection = (
            adjusted_rate * projected_pa
            if config["multiply_by_projected_pa"]
            else adjusted_rate * config["rate_scale"]
        )
        rows.append(
            {
                "player_id": player_id,
                "origin_season": origin_season,
                "metric": metric,
                "marcel_seasons_used": seasons_used,
                "marcel_weighted_pa": weighted_pa,
                "marcel_reliability": weighted_pa / (weighted_pa + regression_pa),
                "marcel_league_baseline": league_value,
                "marcel_league_target": target_num / target_denom,
                "marcel_projected_pa": projected_pa,
                "marcel_age_adjustment": age_adjustment(float(age) + 1.0),
                "marcel_pred_unscaled": projection,
            }
        )

    out = pd.DataFrame(rows)

    if recenter:
        def rescale(block: pd.DataFrame) -> pd.Series:
            aggregate = float(
                np.average(
                    block["marcel_pred_unscaled"], weights=block["marcel_projected_pa"]
                )
            )
            target = float(block["marcel_league_target"].iloc[0])
            multiplier = target / aggregate if aggregate > 0 else 1.0
            return pd.Series(block["marcel_pred_unscaled"] * multiplier, index=block.index)

        out["marcel_pred"] = (
            out.groupby("origin_season", group_keys=False).apply(rescale).astype(float)
        )
    else:
        out["marcel_pred"] = out["marcel_pred_unscaled"]

    assert out["marcel_pred"].notna().all()
    return out
