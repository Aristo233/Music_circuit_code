"""Causal validation: ablation and enhancement of selected components.

Both interventions act on every prediction position of a teacher-forced score
(the shared empty-tag history of a lyric):

* ablation    style tag; the selected components are set to zero;
* enhancement empty tag; ``alpha * delta_s`` is added to the selected components.

Readouts, per position, then aggregated with the lyric's own stepwise KL weights
(``delta_s_kl``) and with uniform weights (``delta_s_uniform``):

    Delta S = sum_layers < h~ - h , u_s^(layer) >

read after the sublayer where the intervened components write (after the MLP
for neurons, after attention for heads), and the output-level effect

    ablation:    KL(prompt || ablated)  / KL(prompt || blank)
    enhancement: 1 - KL(prompt || enhanced) / KL(prompt || blank).

The single-position variant of the paper is the same computation restricted to
one position (``positions=[p]``) with the single-position direction.
"""
from __future__ import annotations

import numpy as np
import torch

from . import capture
from .constants import SITE_OF_KIND, WIDTHS, NUM_HEADS, channels
from .hooks import random_selection

STAGE_VOCAB = {"abc": "abc", "semantic": "semantic"}


class TeacherForcedProbe:
    """One no-cache forward with optional component edits and residual projections."""

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def forward(self, ids, start, unit_directions, stage, edits=None):
        """``unit_directions``: ``[2, layer, hidden]`` FP32 tensor (site 0 attention, 1 MLP).

        ``edits``: ``{kind: {layer: (channel index tensor, "zero"|"add", add vector)}}``
        applied to rows ``start..``. Returns projections ``[2, layer, rows-1]``
        and stage log-probs ``[rows-1, vocab]`` (the last row, which predicts the
        token after the history, is dropped).
        """
        capture.require_clean(self.model)
        layers = self.model.model.layers
        rows = len(ids) - start
        proj = torch.zeros(2, len(layers), rows, device=self.device)
        handles = []

        def edit_hook(kind, layer):
            idx, mode, add = edits[kind][layer]

            def hook(module, args):
                x = args[0].clone()
                if mode == "zero":
                    x[0, start:, idx] = 0
                else:
                    x[0, start:, idx] = x[0, start:, idx] + add.to(x.dtype)
                return (x,) + tuple(args[1:])
            return hook

        def read_attention(layer):
            def hook(m, a):
                proj[0, layer].copy_(a[0][0, start:].float() @ unit_directions[0, layer])
            return hook

        def read_mlp(layer):
            def hook(m, a, o):
                out = o[0] if isinstance(o, tuple) else o
                proj[1, layer].copy_(out[0, start:].float() @ unit_directions[1, layer])
            return hook

        try:
            for layer, block in enumerate(layers):
                handles.append(block.post_attention_layernorm.register_forward_pre_hook(read_attention(layer)))
                handles.append(block.register_forward_hook(read_mlp(layer)))
                if edits:
                    if layer in edits.get("mlp", {}):
                        handles.append(block.mlp.down_proj.register_forward_pre_hook(edit_hook("mlp", layer)))
                    if layer in edits.get("attention", {}):
                        handles.append(block.self_attn.o_proj.register_forward_pre_hook(edit_hook("attention", layer)))
            logits = self.model(torch.tensor([list(ids)], device=self.device), use_cache=False,
                                logits_to_keep=rows).logits[0, :-1]
        finally:
            for h in handles:
                h.remove()
        logp = capture.stage_logits(logits, stage).float().log_softmax(-1)
        return proj[:, :, :-1], logp


def build_edits(probe, kind, selection, mode, delta=None, alpha=1.0, layer_scale=None):
    out = {}
    for layer, units in selection.items():
        if not units:
            continue
        idx = torch.as_tensor(channels(kind, units), device=probe.device)
        add = None
        if mode == "add":
            vec = np.asarray(delta[layer], dtype=np.float64)[channels(kind, units)] * alpha
            if layer_scale is not None:
                vec = vec * layer_scale[layer]
            add = torch.as_tensor(vec, device=probe.device, dtype=torch.float32)
        out[layer] = (idx, mode, add)
    return {kind: out}


def layer_norms(kind, selection, delta):
    return {l: float(np.linalg.norm(np.asarray(delta[l])[channels(kind, u)])) for l, u in selection.items() if u}


def validate_lyric(probe, tokenizer, lyrics, abc_ids, style_text, unit, delta, selections_by_budget, *,
                   kind, alpha=1.0, random_seeds=(1, 2, 3), disjoint=True, seed=0, stage="abc"):
    """Ablation and enhancement curves of one lyric and style.

    ``unit``: ``[2, layer, hidden]`` unit direction of the style (KL-weighted
    u_s, or a single-position direction broadcast over positions);
    ``delta``: ``[layer, width]`` component difference of ``kind``;
    ``selections_by_budget``: ``{budget: {layer: ids}}`` target circuits.
    Yields result rows with ``delta_s_kl``, ``delta_s_uniform`` and ``kl_effect``
    for the target and for random controls (plus norm-matched random controls
    for enhancement).
    """
    prefixes = capture.matched_prefixes(tokenizer, lyrics, style_text, stage, seed=seed)
    blank, target = prefixes["blank"], prefixes["target"]
    history = list(abc_ids)
    u = torch.as_tensor(unit, device=probe.device, dtype=torch.float32)
    base_b, lp_b = probe.forward(blank + history, len(blank) - 1, u, stage)
    base_t, lp_t = probe.forward(target + history, len(target) - 1, u, stage)
    kl_pb = capture.kl(lp_t, lp_b)
    w = (kl_pb / kl_pb.sum()).cpu().numpy()
    total = float(kl_pb.sum())
    site = SITE_OF_KIND[kind]
    width = WIDTHS["mlp"] if kind == "mlp" else NUM_HEADS

    def readout(proj, base, lp):
        ds = (proj - base)[site].sum(0).cpu().numpy()
        k = float(capture.kl(lp_t, lp).sum())
        return dict(delta_s_kl=float(ds @ w), delta_s_uniform=float(ds.mean()), kl_to_prompt=k)

    for budget, selection in selections_by_budget.items():
        actual = sum(len(v) for v in selection.values())
        controls = [("target", selection)]
        for rs in random_seeds:
            controls.append((f"random_{rs}", random_selection({kind: selection}, seed=rs, widths={kind: width}, disjoint=disjoint)[kind]))
        for name, sel in controls:
            if actual == 0 and name != "target":
                continue
            proj, lp = (probe.forward(target + history, len(target) - 1, u, stage, build_edits(probe, kind, sel, "zero"))
                        if actual else (base_t, lp_t))
            r = readout(proj, base_t, lp)
            yield dict(mode="ablation", control=name, budget=budget, actual=actual, kl_total=total,
                       kl_effect=r["kl_to_prompt"] / total, delta_s_kl=r["delta_s_kl"], delta_s_uniform=r["delta_s_uniform"])
            variants = [(name, None)]
            if name.startswith("random") and actual:
                tn, rn = layer_norms(kind, selection, delta), layer_norms(kind, sel, delta)
                variants.append((name.replace("random", "random_norm"), {l: (tn[l] / rn[l] if rn.get(l) else 0.0) for l in rn}))
            for label, scale in variants:
                proj, lp = (probe.forward(blank + history, len(blank) - 1, u, stage, build_edits(probe, kind, sel, "add", delta, alpha, scale))
                            if actual else (base_b, lp_b))
                r = readout(proj, base_b, lp)
                yield dict(mode="enhancement", control=label, budget=budget, actual=actual, kl_total=total,
                           kl_effect=1 - r["kl_to_prompt"] / total, delta_s_kl=r["delta_s_kl"], delta_s_uniform=r["delta_s_uniform"])
    torch.cuda.empty_cache()
