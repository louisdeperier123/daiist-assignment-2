"""
Cost model for the spray / review / don't-spray decision.

Shared by train.py (to sweep thresholds and pick the operating point) and
app.py (to recompute everything live when the sliders move), so both always
use the same formulas and the same euro figures.

Unit of decision: one plant (or one false detection on bare soil).
Positive class: WEED, meaning "spray".

Every euro figure below is an assumption, justified in REPORT.md.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    # Error costs, euros per plant
    fp_crop_sprayed: float = 0.020   # crop plant killed by non-selective herbicide
    fn_weed_missed: float = 0.010    # weed left to compete and set seed
    tp_weed_sprayed: float = 0.001   # one spot dose of herbicide
    tn_crop_left: float = 0.0        # correct, costs nothing
    soil_sprayed: float = 0.001      # dose wasted on a detection with no plant

    # Human review
    review_seconds: float = 2.5      # one plant checked on a tablet
    reviewer_eur_per_hour: float = 20.0

    # Production assumptions
    base_rate_weed: float = 0.40     # share of plants that are weeds in a real field
    hectares_per_day: float = 20.0   # area surveyed by drone per day
    crop_plants_per_ha: float = 80_000.0  # maize planting density
    reviewers_available: int = 1
    review_hours_per_day: float = 8.0

    @property
    def review_cost(self) -> float:
        return self.review_seconds / 3600.0 * self.reviewer_eur_per_hour

    @property
    def plants_per_day(self) -> float:
        # crop plants are fixed by density; weeds scale with the base rate
        return self.hectares_per_day * self.crop_plants_per_ha / (1.0 - self.base_rate_weed)

    @property
    def review_capacity_per_day(self) -> float:
        return self.reviewers_available * self.review_hours_per_day * 3600.0 / self.review_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["review_cost"] = self.review_cost
        return d


def outcome_rates(weed_scores, crop_scores, soil_scores, n_plants, t_low, t_high):
    """Share of each class sent to spray / review / no-spray.

    score >= t_high          -> spray
    t_low <= score < t_high  -> human review
    score <  t_low           -> don't spray
    Plants with no matching weed detection have score 0.
    Soil rates are false detections per plant in the test set.
    """
    t_low = min(t_low, t_high)
    w, c, s = (np.asarray(x, dtype=float) for x in (weed_scores, crop_scores, soil_scores))

    def split(x):
        n = max(len(x), 1)
        spray = np.sum(x >= t_high) / n
        review = np.sum((x >= t_low) & (x < t_high)) / n
        return spray, review, 1.0 - spray - review

    ws, wr, wn = split(w)
    cs, cr, cn = split(c)
    n_plants = max(n_plants, 1)
    ss = np.sum(s >= t_high) / n_plants
    sr = np.sum((s >= t_low) & (s < t_high)) / n_plants
    return dict(weed=(ws, wr, wn), crop=(cs, cr, cn), soil=(ss, sr))


def expected_cost_per_plant(rates, costs: Costs, human_accuracy: float = 1.0):
    """Expected euros per plant, re-weighted to the assumed production base rate."""
    p = costs.base_rate_weed
    ws, wr, wn = rates["weed"]
    cs, cr, cn = rates["crop"]
    ss, sr = rates["soil"]
    rc = costs.review_cost
    h = human_accuracy

    weed = (ws * costs.tp_weed_sprayed
            + wr * (rc + h * costs.tp_weed_sprayed + (1 - h) * costs.fn_weed_missed)
            + wn * costs.fn_weed_missed)
    crop = (cs * costs.fp_crop_sprayed
            + cr * (rc + h * costs.tn_crop_left + (1 - h) * costs.fp_crop_sprayed)
            + cn * costs.tn_crop_left)
    soil = ss * costs.soil_sprayed + sr * rc
    return p * weed + (1 - p) * crop + soil


def review_share(rates, costs: Costs):
    """Reviews per plant (plants in the band plus soil detections in the band)."""
    p = costs.base_rate_weed
    return p * rates["weed"][1] + (1 - p) * rates["crop"][1] + rates["soil"][1]


def confusion_per_1000(rates, costs: Costs):
    """3x3 table per 1,000 plants: rows actual (weed, crop, soil), cols decision."""
    p = costs.base_rate_weed
    ws, wr, wn = rates["weed"]
    cs, cr, cn = rates["crop"]
    ss, sr = rates["soil"]
    return np.array([
        [1000 * p * ws, 1000 * p * wr, 1000 * p * wn],
        [1000 * (1 - p) * cs, 1000 * (1 - p) * cr, 1000 * (1 - p) * cn],
        [1000 * ss, 1000 * sr, 0.0],
    ])


def reference_policies(costs: Costs):
    """Cost per plant of the policies the model is priced against."""
    p = costs.base_rate_weed
    return {
        "Do nothing (never spray)": p * costs.fn_weed_missed,
        "Spray every plant": p * costs.tp_weed_sprayed + (1 - p) * costs.fp_crop_sprayed,
        "Person checks every plant": costs.review_cost + p * costs.tp_weed_sprayed,
    }
