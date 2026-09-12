"""Development sensor script selecting among complete qualified motion schedules.

This baseline uses the common compact corridor measurements. Its thresholds are
development settings, not physical guarantees or privileged beam estimates.
"""

import numpy as np

BANDS = ((0, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4))
DEFAULT_CONFIG = {
    "minimum_observed_free_height_m": 1.36,
    "immediate_prior_band_count": 1,
    "later_prior_band_count": 2,
    "sustained_minimum_hazard_bands": 2,
}


def choose_sensor_schedule(names, features, legality, tick, bank, config=None):
    """Wait or select a legal prior/short/sustained schedule using sensor bands.

    An observed floor-ceiling gap can dismiss an upper hit as sufficiently high.
    Without both surfaces the upper hit remains a conservative hazard cue.
    There is no stopping command, range-to-beam identity, or hidden outcome input.
    """
    settings = dict(DEFAULT_CONFIG if config is None else config)
    legal = np.asarray(legality)
    values = np.asarray(features, dtype=float)
    if (
        set(settings) != set(DEFAULT_CONFIG)
        or not np.isfinite(settings["minimum_observed_free_height_m"])
        or settings["minimum_observed_free_height_m"] <= 0
        or any(
            type(settings[key]) is not int or not 1 <= settings[key] <= len(BANDS)
            for key in DEFAULT_CONFIG
            if key != "minimum_observed_free_height_m"
        )
        or values.shape != (len(names),)
        or len(set(names)) != len(names)
        or not np.isfinite(values).all()
        or legal.shape != (len(bank.option_ids),)
        or legal.dtype.kind != "b"
        or not legal[0]
    ):
        raise ValueError(
            "explicit neutral legal state, finite named features and script settings required"
        )
    observed = dict(zip(names, values, strict=True))
    hazards = []
    for start, end in BANDS:
        prefix = f"corridor_{start:g}_{end:g}_"
        floor_seen = observed[prefix + "floor_fraction"] > 0
        ceiling_seen = observed[prefix + "ceiling_fraction"] > 0
        gap = observed[prefix + "minimum_ceiling_m"] - observed[prefix + "maximum_floor_m"]
        upper = observed[prefix + "upper_hit"] > 0
        hazards.append(
            bool(
                upper
                and not (
                    floor_seen
                    and ceiling_seen
                    and gap >= settings["minimum_observed_free_height_m"]
                )
            )
        )
    phases = sorted({option["entry_tick"] for option in bank.request["options"]})
    if tick not in phases:
        raise ValueError("script is evaluated only at registered neutral decision ticks")
    qualified = {}
    for option in bank.request["options"]:
        index = bank.option_ids.index(option["option_id"])
        if option["entry_tick"] == tick and legal[index]:
            qualified.setdefault(option["reference_id"], []).append(option["option_id"])
    preferred = None
    if tick == phases[0]:
        if any(hazards[: settings["immediate_prior_band_count"]]):
            preferred = "prior_splice"
    elif tick != phases[-1]:
        if any(hazards[: settings["later_prior_band_count"]]):
            preferred = "prior_splice"
    elif any(hazards):
        preferred = (
            "sustained" if sum(hazards) >= settings["sustained_minimum_hazard_bands"] else "short"
        )
    candidates = qualified.get(preferred, [])
    # A requested reference with no currently legal schedule is not substituted
    # by an unverified stopping/slowing behavior or a late command.
    selected = candidates[0] if candidates else "neutral"
    return selected, dict(
        hazard_bands=hazards,
        preferred_reference=preferred,
        selected_option_id=selected,
        settings=settings,
        scope="development sensor thresholds; actual full-course execution required",
    )
