"""Stage-wise tag ablation (paper, "Stage localization").

Each of the three stages (score, music tokens, acoustic rendering) takes the
style text empty (0) or the style prompt (1): eight arms ``a{abc}s{music}n{nar}``.
Outputs of earlier stages are reused when a later stage's tag is switched, so
arms sharing the score condition share one score and arms sharing the score and
music-token conditions share one music-token sequence. The CLAP-margin gain of
the full tag over the empty tag is split among the stages by Shapley values,
which sum exactly to the total gain.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .constants import LABELS
from .steering import recipe

STAGES = ("abc", "semantic", "nar")
ARMS = tuple(f"a{a}s{s}n{n}" for a, s, n in itertools.product((0, 1), repeat=3))
FULL, BLANK = "a1s1n1", "a0s0n0"


def bits(arm):
    return dict(zip(STAGES, (int(arm[1]), int(arm[3]), int(arm[5]))))


def arm_name(on):
    return "a{}s{}n{}".format(*(int(on[s]) for s in STAGES))


def factorial_recipes(style):
    """``{arm: recipe}``: no circuit anywhere, only the stage prompts differ."""
    label = LABELS[style]
    return {arm: recipe(**{s + "_prompt": label if on else "" for s, on in bits(arm).items()}) for arm in ARMS}


def shapley_weights(stage):
    """Weights over arms whose weighted margin sum is the Shapley value of ``stage``."""
    others = [s for s in STAGES if s != stage]
    weights = {a: 0.0 for a in ARMS}
    for size in range(len(others) + 1):
        coef = math.factorial(size) * math.factorial(len(STAGES) - size - 1) / math.factorial(len(STAGES))
        for subset in itertools.combinations(others, size):
            on = {s: int(s in subset) for s in STAGES}
            weights[arm_name({**on, stage: 1})] += coef
            weights[arm_name({**on, stage: 0})] -= coef
    return weights


def main_effect_weights(stage):
    """Average of the four simple effects of ``stage`` (other stages held fixed)."""
    return {a: (1 if bits(a)[stage] else -1) / 4 for a in ARMS}


def contrast(margins, weights):
    """``margins``: ``{arm: margin}`` for one (lyric, style); returns the weighted sum, or None if an arm is missing."""
    if any(a not in margins or margins[a] is None for a, w in weights.items() if w):
        return None
    return float(sum(w * margins[a] for a, w in weights.items()))


def decompose(rows):
    """``rows``: iterable of ``{"song_id", "style", "arm", "margin"}``.

    Returns per style the mean Shapley value of each stage, its share of the
    full-tag effect, the main effects and the number of complete pairs.
    """
    table = {}
    for r in rows:
        table.setdefault((r["song_id"], r["style"]), {})[r["arm"]] = r["margin"]
    out = {}
    for (song, style), margins in table.items():
        cell = out.setdefault(style, {"shapley": {s: [] for s in STAGES}, "main": {s: [] for s in STAGES}, "total": []})
        total = contrast(margins, {FULL: 1.0, BLANK: -1.0})
        if total is None:
            continue
        cell["total"].append(total)
        for stage in STAGES:
            cell["shapley"][stage].append(contrast(margins, shapley_weights(stage)))
            cell["main"][stage].append(contrast(margins, main_effect_weights(stage)))
    summary = {}
    for style, cell in out.items():
        total = float(np.mean(cell["total"])) if cell["total"] else float("nan")
        summary[style] = dict(pairs=len(cell["total"]), full_tag_effect=total,
                              shapley={s: float(np.mean(v)) for s, v in cell["shapley"].items()},
                              share={s: float(np.mean(v)) / total if total else float("nan") for s, v in cell["shapley"].items()},
                              main_effect={s: float(np.mean(v)) for s, v in cell["main"].items()})
    return summary
