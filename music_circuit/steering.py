"""Generating a song under a style circuit, and the conventional steering baselines.

The final recipe of the paper ("dynamic circuit"):

* every style text is empty in all three stages;
* score stage: the KL-weighted component difference ``delta_s`` of the style,
  scaled by ``alpha = 0.5``, is added at every score step to the 392 neurons and
  168 heads selected for the style;
* music-token stage: the difference averaged with equal weights over the first
  128 music-token steps is added at every step to the music-token circuit,
  per-layer norm matched to the single-position circuit's direction, with
  strength 1.0 (1.5 for classic funk and electric blues);
* acoustic rendering (NAR) is not intervened on.

Baselines: the empty tag (no hooks), the style prompt (style text in the
stages, no hooks), a random circuit (same per-layer counts, disjoint from the
target, injection norm matched per layer) and CAA, which adds the mean residual
difference after one decoder layer at every score step and is expressed in
component space so that the same hook runs it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import LABELS, STYLES, WIDTHS, HEAD_DIM, canonical_selection, channels, masked_delta
from .hooks import stage_intervention, match_selected_norms, random_selection, validate_coverage

MUSIC_GAINS = {"pop_ballad": 1.0, "metal_punk": 1.0, "funk": 1.5, "jazz_blues": 1.5, "country_folk": 1.0}
ABC_GAIN = 0.5


# ----------------------------------------------------------------------------- recipes

def recipe(**changes):
    """Stage configuration of one generation.

    ``{stage}_prompt`` is the style text of a stage (empty = no style text);
    ``{stage}_delta`` is ``{kind: [layer, width]}`` to inject on ``{stage}_selection``
    with ``{stage}_gain``. Prompt and circuit are never mixed in one stage.
    """
    base = dict(abc_prompt="", semantic_prompt="", nar_prompt="",
                abc_delta=None, abc_selection=None, abc_gain=0.0, abc_first_n=None,
                semantic_delta=None, semantic_selection=None, semantic_gain=0.0, semantic_first_n=None)
    base.update(changes)
    for stage in ("abc", "semantic"):
        if base[stage + "_prompt"] and base[stage + "_gain"]:
            raise ValueError("Prompt and circuit cannot be mixed within a stage")
    return base


def circuit_recipe(style, klw_bank, klw_circuits, music_bank, music_circuits, music_reference, *,
                   abc_gain=ABC_GAIN, music_gains=MUSIC_GAINS):
    """The paper's dynamic-circuit recipe for one style.

    ``klw_bank``/``klw_circuits``: output of ``directions.fit_klw_directions`` /
    ``select_klw_circuits``. ``music_bank``: the same fit with
    ``uniform_window=128`` on music-token histories (stage "semantic");
    ``music_circuits``: the single-position (Music Start) circuits of
    ``importance.allocate_circuit``; ``music_reference``: the single-position
    component deltas used for per-layer norm matching.
    """
    si = list(klw_bank["styles"]).index(style)
    abc_delta = {k: klw_bank["component_delta_" + k][si] for k in ("mlp", "attention")}
    mi = list(music_bank["styles"]).index(style)
    raw = {k: music_bank["component_delta_" + k][mi] for k in ("mlp", "attention")}
    ref = {k: music_reference["component_delta_" + k][mi] for k in ("mlp", "attention")}
    matched, _ = match_selected_norms(raw, ref, music_circuits[style])
    return recipe(abc_delta=masked_delta(abc_delta, klw_circuits[style]), abc_selection=klw_circuits[style], abc_gain=abc_gain,
                  semantic_delta=masked_delta(matched, music_circuits[style]), semantic_selection=music_circuits[style],
                  semantic_gain=music_gains[style])


def prompt_recipe(style, *, stages=("abc", "semantic", "nar")):
    return recipe(**{s + "_prompt": LABELS[style] for s in stages})


def random_circuit_recipe(base, *, seed):
    """Same per-layer counts and injection norm as ``base``, components disjoint from it (score stage)."""
    out = dict(base)
    for stage in ("abc", "semantic"):
        if not base[stage + "_gain"]:
            continue
        target = canonical_selection(base[stage + "_selection"])
        chosen = random_selection(target, seed=seed, widths={"mlp": WIDTHS["mlp"], "attention": WIDTHS["attention"] // HEAD_DIM})
        raw = base[stage + "_delta"]
        adjusted, _ = match_selected_norms(raw, raw, chosen, target_selection=target)
        out[stage + "_selection"], out[stage + "_delta"] = chosen, masked_delta(adjusted, chosen)
    return out


def caa_recipe(base, residual_delta, layer, *, weights, circuit_norm, scale=1.0):
    """CAA baseline: add a residual vector after decoder ``layer`` at every score step.

    The MLP output enters the residual linearly, so adding ``v`` after the layer
    equals adding ``delta = W^T (W W^T)^{-1} v`` to all inputs of that layer's
    ``down_proj``. ``v`` is rescaled to ``scale * circuit_norm``, the norm of the
    circuit's total residual write (see :func:`circuit_residual_write_norm`).
    ``residual_delta``: ``[hidden]`` mean residual difference after the MLP of
    ``layer`` (uniform mean over the first 128 score steps for CAA proper, or
    the KL-weighted one for the ``klwres`` variant). Music-token stage as ``base``.
    """
    w = np.asarray(weights[layer][1], dtype=np.float64)          # down_proj [hidden, 6144]
    v = np.asarray(residual_delta, dtype=np.float64)
    v = v / np.linalg.norm(v) * circuit_norm * scale
    delta = w.T @ np.linalg.solve(w @ w.T, v)
    mlp_width, att_width = w.shape[1], np.asarray(weights[layer][0]).shape[1]
    mlp = np.zeros((len(weights), mlp_width))
    mlp[layer] = delta
    selection = {"mlp": {layer: list(range(mlp_width))}, "attention": {}}
    return dict(base, abc_prompt="", abc_delta={"mlp": mlp, "attention": np.zeros((len(weights), att_width))},
                abc_selection=selection, abc_gain=1.0, abc_first_n=None)


def circuit_residual_write_norm(delta, selection, weights, *, gain=ABC_GAIN):
    """L2 norm of ``gain * sum_{j in J} W[:, j] delta_j`` over the circuit (all layers summed)."""
    total = np.zeros(weights[0][0].shape[0])
    for layer, (o_w, down_w) in weights.items():
        for u in selection["mlp"].get(layer, []):
            total += down_w[:, u].astype(np.float64) * delta["mlp"][layer, u]
        for h in selection["attention"].get(layer, []):
            idx = channels("attention", [h])
            total += o_w[:, idx].astype(np.float64) @ delta["attention"][layer, idx]
    return float(np.linalg.norm(gain * total))


# ----------------------------------------------------------------------------- generation

def _context(pipe, r, stage):
    if not r[stage + "_gain"]:
        return stage_intervention(pipe._load_model(), {"mlp": {}, "attention": {}}, gain=0.0)
    return stage_intervention(pipe._load_model(), r[stage + "_selection"], r[stage + "_delta"],
                              gain=r[stage + "_gain"], first_n=r[stage + "_first_n"])


def generate_song(pipe, lyrics, r, *, abc_seed, music_seed, song_id="song", cot="melody"):
    """Run the three YuE2 stages under recipe ``r``; returns tokens, plan, audio and audits.

    Stage hooks wrap only the two autoregressive calls; no hook is present
    during acoustic rendering. The waveform is 48 kHz stereo float32.
    """
    from yue2 import SemanticResult, SymbolicPlan
    from yue2.protocol import SongRequest, token_prefixes
    tokenizer = pipe.tokenizer
    # score stage
    request = SongRequest(style=r["abc_prompt"], lyrics=lyrics, cot=cot, seed=int(abc_seed), cfg_scale=1.0, id=song_id)
    trace = []
    with _context(pipe, r, "abc") as audit:
        plan = pipe.plan(request=request, on_token=lambda phase, token: trace.append(int(token)))
    validate_coverage(audit, len(trace))
    abc_audit = dict(audit, tokens=len(plan.abc_ids), truncated=bool(plan.truncated))
    # music-token stage: same score, possibly different style text
    request = SongRequest(style=r["semantic_prompt"], lyrics=lyrics, cot=cot, seed=int(music_seed), cfg_scale=1.0, id=song_id)
    music_plan = SymbolicPlan(request, plan.abc, list(plan.abc_ids), token_prefixes(request, tokenizer, plan.abc_ids),
                              dict(plan.timing), plan.truncated)
    trace = []
    with _context(pipe, r, "semantic") as audit:
        semantic = pipe.generate_semantic(music_plan, on_token=lambda phase, token: trace.append(int(token)))
    validate_coverage(audit, len(trace))
    music_audit = dict(audit, tokens=len(semantic.tokens), truncated=bool(semantic.truncated))
    # acoustic rendering: no hooks
    request = replace(request, style=r["nar_prompt"])
    render_plan = SymbolicPlan(request, plan.abc, list(plan.abc_ids), token_prefixes(request, tokenizer, plan.abc_ids),
                               dict(plan.timing), plan.truncated)
    latents = pipe.synthesize(SemanticResult(render_plan, list(semantic.tokens), semantic.timing, semantic.truncated))
    waveform = pipe.decode(latents)
    return dict(abc=plan.abc, abc_ids=list(plan.abc_ids), semantic_tokens=list(semantic.tokens), waveform=waveform,
                sample_rate=48000, complete=not (plan.truncated or semantic.truncated),
                audits={"abc": abc_audit, "semantic": music_audit})
