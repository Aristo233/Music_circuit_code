"""Style directions, component attribution and circuit selection.

Two families of directions are used in the paper.

**Single-position directions** (static observation, Eq. 1--2). At one score or
music-token position, the five style versions of the same lyrics are captured,
the within-lyric mean is subtracted (which cancels shared content), and the
result is averaged over lyrics and normalized. Components are scored by
``activation * <W[:, j], d>`` (neurons) or the head-summed equivalent, and
selected per layer with the sublayer-importance budget (see ``importance``).

**KL-weighted directions** (dynamic circuit steering, Eq. KLW). For every lyric
one score written under the empty tag is the shared history; the style tag and
the empty tag are each run once over the whole history. At every score position
the activation difference is weighted by the stepwise KL between the two
next-token distributions, normalized within the lyric, then averaged over
lyrics. The same weighted average of the residual stream, normalized, is the
unit direction u_s of each sublayer output. Components are scored by the
injected quantity itself, ``delta_j * <W[:, j], u>``, and the positive top
392 neurons / 168 heads are kept, ranking all layers together.
"""
from __future__ import annotations

import numpy as np
import torch

from . import capture
from .constants import BUDGETS, NUM_HEADS, SITE_OF_KIND, STYLES

KINDS = ("attention", "mlp")


# ----------------------------------------------------------------------------- shared math

def normalized(value, epsilon=1e-12):
    value = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.divide(value, norms, out=np.zeros_like(value), where=norms > epsilon)


def grouped_contrast(value):
    """Mean within-lyric style contrast of ``[lyric, style, ...]`` captures (Eq. 1 before normalization)."""
    value = np.asarray(value, dtype=np.float64)
    return (value - value.mean(axis=1, keepdims=True)).mean(axis=0)


def write_scores(component, direction, weight, *, num_heads=None):
    """Signed local write of a component quantity on a unit residual direction.

    ``component``: ``[..., input]`` (mean activation for single-position scores,
    KL-weighted difference for steering scores); ``direction``: ``[..., output]``;
    ``weight``: the PyTorch Linear matrix ``[output, input]`` of ``down_proj`` or
    ``o_proj``. Per channel ``component[j] * <W[:, j], direction>``; with
    ``num_heads`` the channels of each head are summed after multiplication.
    """
    component = np.asarray(component, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    score = component * (direction @ weight)
    if num_heads is not None:
        score = score.reshape(*score.shape[:-1], num_heads, -1).sum(-1)
    return score


def positive_global_selection(scores, budget):
    """Top ``budget`` components with positive score, ranking all layers together.

    ``scores`` is ``[layer, component]``; the result is ``{layer: [ids]}``.
    Fewer positive scores give a smaller circuit; zero or negative scores never
    fill the budget.
    """
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores.ravel(), kind="stable")
    order = order[scores.ravel()[order] > 0][:int(budget)]
    width = scores.shape[1]
    return {int(l): sorted(int(i % width) for i in order if i // width == l)
            for l in range(scores.shape[0]) if np.any(order // width == l)}


def projection_weights(model):
    """``{layer: (o_proj weight, down_proj weight)}`` as FP32 numpy arrays."""
    return {l: (block.self_attn.o_proj.weight.detach().float().cpu().numpy(),
                block.mlp.down_proj.weight.detach().float().cpu().numpy())
            for l, block in enumerate(model.model.layers)}


# ----------------------------------------------------------------------------- single position

def fit_single_position(cap, weights, *, num_heads=NUM_HEADS):
    """Single-position directions and component attribution (paper, Sec. "Static observation").

    ``cap[key]`` has axes ``[lyric, style, layer, channel]`` for one position;
    every lyric contributes all five style versions. Returns

    * ``residual_directions [style, site, layer, hidden]`` (Eq. 1),
    * ``component_delta_{mlp,attention} [style, layer, width]`` (the centered
      activation difference used for enhancement),
    * ``mlp_scores [style, layer, 6144]``, ``head_scores [style, layer, heads]``
      (Eq. 2, scored with the uncentered mean activation of the style),
    * ``sigma [site, layer]``: RMS of the full residual, for sublayer importance.
    """
    songs, styles, layers, hidden = cap["res_attention"].shape
    out = {"residual_directions": np.empty((styles, 2, layers, hidden)),
           "component_delta_mlp": np.empty((styles, layers, cap["gated_mlp"].shape[-1])),
           "component_delta_attention": np.empty((styles, layers, cap["head_concat"].shape[-1])),
           "mlp_scores": np.empty((styles, layers, cap["gated_mlp"].shape[-1])),
           "head_scores": np.empty((styles, layers, num_heads)),
           "sigma": np.empty((2, layers))}
    for layer in range(layers):
        o_w, down_w = weights[layer]
        for site, kind in enumerate(KINDS):
            residual = np.asarray(cap["res_" + kind][:, :, layer], dtype=np.float64)
            contrast = grouped_contrast(residual)
            contrast -= contrast.mean(axis=0)          # exact zero style mean up to roundoff
            direction = normalized(contrast)
            out["residual_directions"][:, site, layer] = direction
            out["sigma"][site, layer] = np.sqrt((residual ** 2).mean(-1)).mean()
            activation = np.asarray(cap["gated_mlp" if kind == "mlp" else "head_concat"][:, :, layer], dtype=np.float64)
            out["component_delta_" + kind][:, layer] = grouped_contrast(activation)
            out["mlp_scores" if kind == "mlp" else "head_scores"][:, layer] = write_scores(
                activation.mean(axis=0), direction, down_w if kind == "mlp" else o_w,
                num_heads=None if kind == "mlp" else num_heads)
    return out


def cross_position_agreement(directions_a, directions_b, selection_a, selection_b):
    """Cosine of two directions and Jaccard overlap of two selections (paper, "Cross-position comparison")."""
    a, b = np.asarray(directions_a).ravel(), np.asarray(directions_b).ravel()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    jaccard = {}
    for kind in ("mlp", "attention"):
        sa = {(l, u) for l, ids in selection_a.get(kind, {}).items() for u in ids}
        sb = {(l, u) for l, ids in selection_b.get(kind, {}).items() for u in ids}
        jaccard[kind] = len(sa & sb) / len(sa | sb) if sa | sb else float("nan")
    return dict(cosine=cosine, jaccard=jaccard)


# ----------------------------------------------------------------------------- KL-weighted

def lyric_klw_deltas(model, tokenizer, lyrics, abc_ids, style_texts, *, uniform_window=None, seed=0):
    """Per style: KL-weighted (or uniform-window) activation difference of one lyric.

    Both branches are teacher-forced over ``prefix + abc_ids``; rows are the
    positions that predict score tokens 1..n. With ``uniform_window=w`` the
    first ``w`` positions are averaged with equal weights instead (this is the
    music-token-stage variant: "mean over the first 128 steps").
    Returns ``({style: {key: [layer, width] float64}}, {style: KL profile})``.
    """
    keys = capture.CAPTURE_KEYS
    n = len(abc_ids) if uniform_window is None else min(int(uniform_window), len(abc_ids))
    blank_prefix = capture.matched_prefixes(tokenizer, lyrics, next(iter(style_texts.values())), "abc", seed=seed)["blank"]
    blank, logp_b = capture.capture_sequence(model, blank_prefix + list(abc_ids), len(blank_prefix) - 1, "abc")
    deltas, profiles = {}, {}
    for style, text in style_texts.items():
        target = capture.matched_prefixes(tokenizer, lyrics, text, "abc", seed=seed)["target"]
        acts, logp_t = capture.capture_sequence(model, target + list(abc_ids), len(target) - 1, "abc")
        if uniform_window is None:
            profile = capture.kl(logp_t[:n], logp_b[:n])
            weights = profile / profile.sum()
            profiles[style] = profile.cpu().numpy()
        else:
            weights = torch.full((n,), 1.0 / n, device=logp_t.device)
        deltas[style] = {k: torch.einsum("p,pld->ld", weights, acts[k][:n].float() - blank[k][:n].float()).double().cpu().numpy()
                         for k in keys}
        del acts
        torch.cuda.empty_cache()
    return deltas, profiles


def fit_klw_directions(model, tokenizer, histories, style_texts, *, uniform_window=None, on_progress=None):
    """Eq. KLW over the representation-analysis lyrics.

    ``histories``: iterable of ``{"lyrics": str, "abc_ids": [int], "seed": int}``
    (one empty-tag score per lyric). Returns arrays indexed ``[style, ...]`` in
    the order of ``style_texts``:

    * ``component_delta_mlp [style, layer, 6144]``, ``component_delta_attention [style, layer, 2048]``
      (delta_s in Eq. KLW; this is exactly what steering injects),
    * ``residual_delta [style, site, layer, hidden]`` (rho_s) and
      ``residual_directions`` (u_s = rho_s / ||rho_s||).
    """
    styles = list(style_texts)
    sums, count, kl_summary = None, 0, []
    for index, row in enumerate(histories):
        deltas, profiles = lyric_klw_deltas(model, tokenizer, row["lyrics"], row["abc_ids"], style_texts,
                                            uniform_window=uniform_window, seed=row.get("seed", 0))
        if sums is None:
            sums = {s: {k: np.zeros_like(v) for k, v in deltas[s].items()} for s in styles}
        for s in styles:
            for k in deltas[s]:
                sums[s][k] += deltas[s][k]
            if s in profiles:
                p = profiles[s]
                kl_summary.append(dict(index=index, style=s, positions=len(p), kl_total=float(p.sum())))
        count += 1
        if on_progress:
            on_progress(dict(phase="klw", done=index + 1))
    mean = {s: {k: v / count for k, v in sums[s].items()} for s in styles}
    residual = np.stack([np.stack([mean[s]["res_attention"], mean[s]["res_mlp"]]) for s in styles])
    return {"styles": np.asarray(styles), "lyrics": count, "kl_summary": kl_summary,
            "component_delta_mlp": np.stack([mean[s]["gated_mlp"] for s in styles]),
            "component_delta_attention": np.stack([mean[s]["head_concat"] for s in styles]),
            "residual_delta": residual, "residual_directions": normalized(residual)}


def klw_component_scores(bank, weights, *, num_heads=NUM_HEADS):
    """Steering scores (Appendix "KL-weighted direction and component selection").

    ``bank`` is the output of :func:`fit_klw_directions`. Returns
    ``{"mlp": [style, layer, 6144], "attention": [style, layer, heads]}``.
    """
    unit = bank["residual_directions"]
    scores = {"mlp": [], "attention": []}
    for si in range(len(bank["styles"])):
        mlp, att = [], []
        for layer, (o_w, down_w) in sorted(weights.items()):
            mlp.append(write_scores(bank["component_delta_mlp"][si, layer], unit[si, SITE_OF_KIND["mlp"], layer], down_w))
            att.append(write_scores(bank["component_delta_attention"][si, layer], unit[si, SITE_OF_KIND["attention"], layer],
                                    o_w, num_heads=num_heads))
        scores["mlp"].append(np.stack(mlp))
        scores["attention"].append(np.stack(att))
    return {k: np.stack(v) for k, v in scores.items()}


def select_klw_circuits(scores, styles, *, budgets=BUDGETS):
    """``{style: {"mlp": {layer: ids}, "attention": {layer: ids}}}`` with positive global ranking."""
    return {style: {kind: positive_global_selection(scores[kind][si], budgets[kind]) for kind in ("mlp", "attention")}
            for si, style in enumerate(styles)}
