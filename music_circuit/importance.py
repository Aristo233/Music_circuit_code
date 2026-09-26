"""Sublayer importance and budget allocation (Appendix "Implementation details").

A sublayer's importance is the change of the final normed hidden state along a
style reference direction when that sublayer's output is pushed along the
style direction by ``alpha * sigma`` (``sigma``: mean residual RMS at that site
and layer), divided by ``alpha * sigma`` and averaged over ``alphas`` and lyrics.
For the score stage the push is applied at every score position and the
readout is averaged with the lyric's stepwise KL weights; the fraction of the
output KL removed is recorded as a second readout.

The music-token stage allocates the per-layer circuit budget by this
importance: every layer gets a minimum quota (4 neurons, 2 heads), the rest is
distributed in proportion to the positive part of the importance with
largest-remainder rounding, and a layer's extra share is capped at 40% of the
budget. The score stage ranks all layers together instead (see ``directions``).
"""
from __future__ import annotations

import numpy as np
import torch

from . import capture
from .constants import BUDGETS

REFERENCE_LAYERS = (21, 22, 23, 24, 25)
ALPHAS = (0.1, 0.3, 0.5, 0.7)
MINIMUM = {"attention": 2, "mlp": 4}


def reference_direction(unit_directions):
    """Sign-aligned mean of the unit directions at the reference layers, both sites.

    ``unit_directions``: ``[style, site, layer, hidden]``. Returns
    ``(aligned_units, reference [style, hidden])``; the sign flip is only used for
    importance measurement, component deltas keep their sign.
    """
    unit = np.asarray(unit_directions, dtype=np.float64)
    styles, _, _, hidden = unit.shape
    terms = unit[:, :, list(REFERENCE_LAYERS)].reshape(styles, -1, hidden)
    anchor = unit[:, 0, REFERENCE_LAYERS[0]]
    signs = np.where(np.einsum("std,sd->st", terms, anchor) < 0, -1.0, 1.0)
    reference = (terms * signs[..., None]).mean(1)
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    flip = np.where(np.einsum("sld,sd->sl", unit.reshape(styles, -1, hidden), reference) < 0, -1.0, 1.0)
    return unit * flip.reshape(unit.shape[:3])[..., None], reference


class SublayerProbe:
    def __init__(self, model):
        self.model, self.device = model, next(model.parameters()).device

    @torch.inference_mode()
    def forward(self, ids, start, reference, stage, edits=None, residual_rms=False):
        """Batched forward over identical ``ids``; ``edits [B, site, layer, hidden]`` are
        added to the sublayer outputs at rows ``start..``. Row 0 must be zero (the
        unedited run). Returns final-norm projections ``[B, rows-1]``, stage
        logits ``[B, rows-1, vocab]`` and residual RMS ``[2, layer]``."""
        capture.require_clean(self.model)
        layers = self.model.model.layers
        batch = 1 if edits is None else edits.shape[0]
        rows = len(ids) - start
        final = torch.zeros(batch, rows, device=self.device)
        rms = torch.zeros(2, len(layers), device=self.device)
        handles = []
        active = set() if edits is None else {(s, l) for s, l in zip(*torch.nonzero(edits.abs().sum((0, 3))).T.tolist())}

        def add(site, layer):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                out = out.clone()
                out[:, start:] = out[:, start:] + edits[:, site, layer, None, :].to(out.dtype)
                return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            return hook

        def measure(site, layer):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                rms[site, layer] = out[0, start:].float().pow(2).mean(-1).sqrt().mean()
            return hook

        try:
            for layer, block in enumerate(layers):
                if (0, layer) in active:
                    handles.append(block.self_attn.register_forward_hook(add(0, layer)))
                if (1, layer) in active:
                    handles.append(block.mlp.register_forward_hook(add(1, layer)))
                if residual_rms:
                    handles.append(block.post_attention_layernorm.register_forward_pre_hook(
                        lambda m, a, l=layer: rms.__setitem__((0, l), a[0][0, start:].float().pow(2).mean(-1).sqrt().mean())))
                    handles.append(block.register_forward_hook(measure(1, layer)))
            def read_final(m, a, o):
                final.copy_(o[:, start:].float() @ reference)

            handles.append(self.model.model.norm.register_forward_hook(read_final))
            logits = self.model(torch.tensor([list(ids)] * batch, device=self.device), use_cache=False,
                                logits_to_keep=rows).logits[:, :-1]
        finally:
            for h in handles:
                h.remove()
        return final[:, :-1], capture.stage_logits(logits, stage), rms


def sigma_pass(probe, tokenizer, histories, *, seed=0):
    """Mean residual RMS ``[site, layer]`` over score positions of the empty-tag histories."""
    acc = []
    for row in histories:
        blank = capture.matched_prefixes(tokenizer, row["lyrics"], "x", "abc", seed=row.get("seed", seed))["blank"]
        _, _, rms = probe.forward(blank + list(row["abc_ids"]), len(blank) - 1,
                                  torch.zeros(probe.model.config.hidden_size, device=probe.device), "abc", residual_rms=True)
        acc.append(rms.cpu().numpy())
    return np.mean(acc, axis=0)


def sublayer_importance(probe, tokenizer, lyrics, abc_ids, style_texts, aligned_units, reference, sigma,
                        *, alphas=ALPHAS, batch_size=8, seed=0):
    """Score-stage sublayer importance of one lyric: ``values/uniform/kl_removed [style, site, layer, alpha]``."""
    styles = list(style_texts)
    layers = aligned_units.shape[2]
    batch_size = max(2, min(batch_size, 4000 // (len(abc_ids) + 1)))
    values = np.full((len(styles), 2, layers, len(alphas)), np.nan)
    uniform, removed = np.full_like(values, np.nan), np.full_like(values, np.nan)
    for si, style in enumerate(styles):
        prefixes = capture.matched_prefixes(tokenizer, lyrics, style_texts[style], "abc", seed=seed)
        blank, target = prefixes["blank"], prefixes["target"]
        ref = torch.as_tensor(reference[si], device=probe.device, dtype=torch.float32)
        _, logits_t, _ = probe.forward(target + list(abc_ids), len(target) - 1, ref, "abc")
        lp_t = logits_t[0].float().log_softmax(-1)
        del logits_t
        cells = [(site, layer, a) for site in range(2) for layer in range(layers) for a in range(len(alphas))]
        for i in range(0, len(cells), batch_size - 1):
            chunk = cells[i:i + batch_size - 1]
            edits = torch.zeros(len(chunk) + 1, 2, layers, aligned_units.shape[-1], device=probe.device)
            for b, (site, layer, a) in enumerate(chunk, 1):
                edits[b, site, layer] = torch.as_tensor(aligned_units[si, site, layer] * alphas[a] * sigma[site, layer],
                                                        device=probe.device, dtype=torch.float32)
            final, logits, _ = probe.forward(blank + list(abc_ids), len(blank) - 1, ref, "abc", edits)
            per_row = [capture.kl(lp_t, logits[b].float().log_softmax(-1)) for b in range(logits.shape[0])]
            del logits
            k = [float(v.sum()) for v in per_row]
            w = per_row[0] / per_row[0].sum()
            for b, (site, layer, a) in enumerate(chunk, 1):
                change = final[b] - final[0]
                scale = alphas[a] * sigma[site, layer]
                values[si, site, layer, a] = float(change @ w) / scale
                uniform[si, site, layer, a] = float(change.mean()) / scale
                removed[si, site, layer, a] = 1 - k[b] / k[0]
        torch.cuda.empty_cache()
    return values, uniform, removed


# ----------------------------------------------------------------------------- budget allocation

def allocate(importance, total, minimum, capacity, *, max_extra_fraction=0.4):
    """Bounded proportional quotas with deterministic largest remainders.

    Negative importance counts as zero; all-zero eligible weights split the
    remaining slots equally. ``max_extra_fraction`` caps the extra slots of a
    layer at ``floor(fraction * total)``.
    """
    weights = np.asarray(importance, dtype=np.float64)
    lower = np.broadcast_to(minimum, weights.shape).astype(np.int64)
    upper = np.broadcast_to(capacity, weights.shape).astype(np.int64)
    if max_extra_fraction is not None:
        upper = np.minimum(upper, lower + int(np.floor(max_extra_fraction * total)))
    if not int(lower.sum()) <= total <= int(upper.sum()):
        raise ValueError("Budget infeasible under minimum/capacity bounds")
    counts = lower.copy()
    while int(counts.sum()) < total:
        available = upper - counts
        eligible = available > 0
        positive = np.maximum(weights, 0) * eligible
        if not np.any(positive > 0):
            positive = eligible.astype(np.float64)
        positive = positive / positive.max()
        remaining = total - int(counts.sum())
        ideal = remaining * positive / positive.sum()
        floor = np.floor(ideal).astype(np.int64)
        counts += np.minimum(floor, available)
        remaining = total - int(counts.sum())
        order = sorted(np.flatnonzero(counts < upper), key=lambda i: (-(ideal[i] - floor[i]), i))
        counts[np.asarray(order[:remaining], dtype=np.int64)] += 1
    return counts


def allocate_circuit(importance, mlp_scores, head_scores, *, budgets=BUDGETS, minimum=MINIMUM, max_extra_fraction=0.4):
    """Per-layer quotas by importance, then the top components per layer (music-token stage).

    ``importance [site, layer]``, ``mlp_scores [layer, 6144]``, ``head_scores [layer, heads]``
    for one stage and style. Returns ``{"mlp": {layer: ids}, "attention": {layer: ids}}``.
    """
    out = {}
    for site, kind in enumerate(("attention", "mlp")):
        scores = np.asarray(head_scores if kind == "attention" else mlp_scores, dtype=np.float64)
        quota = allocate(importance[site], budgets[kind], minimum[kind], scores.shape[-1], max_extra_fraction=max_extra_fraction)
        out[kind] = {l: np.argsort(-scores[l], kind="stable")[:int(quota[l])].tolist() for l in range(len(quota)) if quota[l]}
    return out
