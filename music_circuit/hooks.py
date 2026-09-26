"""Component interventions on the autoregressive stages of YuE2.

``stage_intervention`` wraps exactly one ``pipe.plan(...)`` (score stage) or
``pipe.generate_semantic(...)`` (music-token stage) call. At the last position of
every AR forward (the final prefill position and each cached decode step) it adds
``gain * delta`` to the selected input channels of ``down_proj`` (MLP neurons)
and ``o_proj`` (attention heads), and leaves everything else unchanged:

    a_J  <-  a_J + alpha * delta_J          (paper, "Steering")

Deltas are FP32 ``[layer, width]`` arrays per kind; ``phase_deltas`` may give a
schedule ``{anchor_step: delta}`` (the delta with the greatest anchor <= step is
used; the paper's final recipe uses a single delta for the whole stage).
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
import math

import numpy as np
import torch

from .constants import canonical_selection, channels, HEAD_DIM

KINDS = ("mlp", "attention")


class UnmatchableNormError(ValueError):
    """A random control whose selected delta is zero cannot be norm-matched."""


def _arrays(deltas, selection, head_dim):
    out = {}
    for kind, layers in selection.items():
        if not layers:
            continue
        if kind not in deltas:
            raise ValueError(f"Missing {kind} deltas")
        value = np.asarray(deltas[kind], dtype=np.float64)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("Deltas must be finite [layer, width] arrays")
        for layer, units in layers.items():
            idx = channels(kind, units, head_dim)
            if layer >= len(value) or max(idx) >= value.shape[1]:
                raise ValueError("Selected layer/component exceeds delta dimensions")
        out[kind] = value.copy()
    return out


def match_selected_norms(new_deltas, reference_deltas, selection, *, head_dim=HEAD_DIM, target_selection=None):
    """Rescale ``new_deltas`` so that, per layer and kind, the L2 norm over the
    selected channels equals the reference's norm over ``target_selection``
    (defaults to the same selection). Used for the norm-matched random control
    and for matching a new direction bank to a released one."""
    chosen = canonical_selection(selection)
    target = chosen if target_selection is None else canonical_selection(target_selection)
    new, ref = _arrays(new_deltas, chosen, head_dim), _arrays(reference_deltas, target, head_dim)
    out, report = {k: v.copy() for k, v in new.items()}, []
    for kind in KINDS:
        if set(chosen[kind]) != set(target[kind]):
            raise ValueError("Control and reference must use the same layers")
        for layer, units in chosen[kind].items():
            src, dst = channels(kind, units, head_dim), channels(kind, target[kind][layer], head_dim)
            have, want = float(np.linalg.norm(new[kind][layer, src])), float(np.linalg.norm(ref[kind][layer, dst]))
            if have == 0 and want > 0:
                raise UnmatchableNormError(f"{kind} layer {layer}: zero source norm")
            scale = want / have if have else 0.0
            out[kind][layer, src] *= scale
            report.append(dict(kind=kind, layer=layer, components=len(units), scale=scale, l2=want))
    return out, report


def random_selection(target, *, seed, widths, disjoint=True):
    """Random components with the target's per-layer counts.

    ``disjoint=True`` (paper, Appendix "Random component controls") draws only
    components the target does not select; where a layer has too few left, the
    shortfall moves to the nearest layer that also contains target components.
    ``widths`` maps kind to the number of components per layer (6144 neurons,
    16 heads).
    """
    target = canonical_selection(target)
    rng = np.random.default_rng(seed)
    out = {"mlp": {}, "attention": {}}
    for kind in KINDS:
        width = widths[kind]
        used = {l: set(u) for l, u in target[kind].items()}
        if not used:
            continue
        if not disjoint:
            out[kind] = {l: sorted(rng.choice(width, size=len(u), replace=False).tolist()) for l, u in used.items()}
            continue
        free = {l: [u for u in range(width) if u not in used[l]] for l in used}
        take = {l: min(len(used[l]), len(free[l])) for l in used}
        for layer in sorted(used, key=lambda l: (-(len(used[l]) - take[l]), l)):
            short = len(used[layer]) - min(len(used[layer]), len(free[layer]))
            for other in sorted(used, key=lambda l: (abs(l - layer), l)):
                if short and other != layer and take[other] < len(free[other]):
                    extra = min(short, len(free[other]) - take[other])
                    take[other] += extra
                    short -= extra
            if short:
                raise ValueError("Not enough non-target components for a disjoint control")
        out[kind] = {l: sorted(rng.choice(free[l], size=take[l], replace=False).tolist()) for l in used if take[l]}
    return out


@contextmanager
def stage_intervention(model, selection, deltas=None, *, gain=1.0, first_n=None, phase_deltas=None):
    """Add ``gain * delta`` to selected components at every AR prediction step.

    A zero gain, an empty selection or ``first_n == 0`` installs no hooks.
    ``first_n`` limits the edit to the first ``n`` prediction steps (step 0 is
    the last prefill position). The yielded dict counts calls per module so a
    caller can check that every step of the stage was covered.
    """
    gain = float(gain)
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    chosen = canonical_selection(selection)
    phases = {0: deltas} if phase_deltas is None else {int(k): v for k, v in phase_deltas.items()}
    if 0 not in phases:
        raise ValueError("A phase schedule must start at step 0")
    anchors = sorted(phases)
    audit = {"gain": gain, "first_n": first_n, "prediction_steps": 0, "modules": {}, "active": True}
    if gain == 0 or first_n == 0 or not any(chosen.values()):
        audit["active"] = False
        yield audit
        return
    if hook_inventory(model):
        raise ValueError("Active intervention requires a model without existing hooks")
    layers = model.model.layers
    head_dim = int(layers[0].self_attn.head_dim)
    arrays = {a: _arrays(v, chosen, head_dim) for a, v in phases.items()}
    prepared = []
    for kind in KINDS:
        for layer, units in sorted(chosen[kind].items()):
            block = layers[layer]
            module = block.mlp.down_proj if kind == "mlp" else block.self_attn.o_proj
            idx = torch.tensor(channels(kind, units, head_dim))
            vectors = {a: torch.as_tensor(arrays[a][kind], dtype=torch.float32)[layer, idx].clone() * gain for a in anchors}
            stats = {"kind": kind, "layer": layer, "units": len(units), "calls": 0, "edited_calls": 0}
            audit["modules"][f"{kind}.{layer}"] = stats
            prepared.append((module, idx, vectors, stats))
    state = {"step": 0}
    handles = []

    def before_backbone(_module, args, kwargs):
        if kwargs.get("ar_mask", args[5] if len(args) > 5 else None) is not None:
            raise RuntimeError("Interventions are AR-only; close the context before acoustic rendering")
        state["step"] = audit["prediction_steps"]
        audit["prediction_steps"] += 1

    def projection_hook(idx, vectors, stats):
        cache = {}

        def hook(_module, args):
            value = args[0]
            stats["calls"] += 1
            step = state["step"]
            if first_n is not None and step >= first_n:
                return None
            anchor = anchors[bisect_right(anchors, step) - 1]
            key = (anchor, value.device, value.dtype)
            if key not in cache:
                cache[key] = (idx.to(value.device), vectors[anchor].to(device=value.device, dtype=value.dtype))
            local_idx, local_delta = cache[key]
            changed = value.clone()
            changed[:, -1, local_idx] += local_delta
            stats["edited_calls"] += 1
            return (changed, *args[1:])
        return hook

    try:
        handles.append(model.model.register_forward_pre_hook(before_backbone, with_kwargs=True))
        for module, idx, vectors, stats in prepared:
            handles.append(module.register_forward_pre_hook(projection_hook(idx, vectors, stats)))
        yield audit
    finally:
        for handle in handles:
            handle.remove()
        if hook_inventory(model):
            raise RuntimeError("Intervention hooks leaked")


def validate_coverage(audit, prediction_steps):
    """Every selected module must have run once per AR prediction step."""
    if not audit["active"]:
        return audit
    if audit["prediction_steps"] != prediction_steps:
        raise ValueError("AR step count differs from the generation callback")
    edited = prediction_steps if audit["first_n"] is None else min(prediction_steps, audit["first_n"])
    for stats in audit["modules"].values():
        if stats["calls"] != prediction_steps or stats["edited_calls"] != edited:
            raise ValueError("Incomplete hook coverage")
    return audit


def hook_inventory(model):
    return {name: (len(m._forward_pre_hooks), len(m._forward_hooks))
            for name, m in model.named_modules() if m._forward_pre_hooks or m._forward_hooks}
