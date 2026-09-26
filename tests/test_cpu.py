"""CPU checks on a tiny YuE2 architecture; no checkpoint, no GPU.

Run from the package root with ``python -m pytest tests`` or plainly with
``python tests/test_cpu.py`` (needs the upstream ``yue2`` package importable,
e.g. ``pip install -e path/to/YuE/upstream``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from music_circuit import capture, causal, directions, evaluation, hooks, importance, stage_ablation, steering  # noqa: E402
from music_circuit.constants import canonical_selection  # noqa: E402

import yue2.modeling_yue2 as yue2  # noqa: E402


def tiny():
    torch.manual_seed(29)
    return yue2.YuE2ForCausalLM(yue2.YuE2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=24, vocab_size=64, latent_dim=4, max_latent_frames=16, max_position_embeddings=64)).eval()


def decode(model, count):
    outputs, cache = [], None
    with torch.no_grad():
        for step in range(count):
            ids = torch.tensor([[2, 4, 7, 5]]) if step == 0 else torch.tensor([[8 + step]])
            row = model(ids, past_key_values=cache, use_cache=True)
            outputs.append(row.logits.detach().clone())
            cache = row.past_key_values
    return outputs


SELECTION = {"mlp": {0: [2, 9]}, "attention": {1: [2]}}
DELTAS = {"mlp": np.arange(48, dtype=np.float64).reshape(2, 24) / 13,
          "attention": np.arange(32, dtype=np.float64).reshape(2, 16) / 17}


def test_intervention_changes_logits_every_step_and_cleans_up():
    model = tiny()
    plain = decode(model, 4)
    with hooks.stage_intervention(model, SELECTION, DELTAS, gain=1.0) as audit:
        edited = decode(model, 4)
    hooks.validate_coverage(audit, 4)
    assert audit["modules"]["mlp.0"]["edited_calls"] == 4
    assert hooks.hook_inventory(model) == {}
    assert any(not torch.equal(a, b) for a, b in zip(plain, edited))
    with hooks.stage_intervention(model, SELECTION, DELTAS, gain=0.0) as audit:
        same = decode(model, 4)
    assert not audit["active"] and all(torch.equal(a, b) for a, b in zip(plain, same))
    with hooks.stage_intervention(model, SELECTION, DELTAS, gain=1.0, first_n=2) as audit:
        limited = decode(model, 4)
    hooks.validate_coverage(audit, 4)
    assert audit["modules"]["mlp.0"]["edited_calls"] == 2


def test_capture_sites_match_manual_hooks():
    model = tiny()
    ids = [2, 4, 7, 5, 9, 11]
    acts = capture.capture_positions(model, ids, [len(ids) - 1])
    assert acts["gated_mlp"].shape == (2, 24) and acts["head_concat"].shape == (2, 16)
    assert acts["res_mlp"].shape == (2, 16)
    seq, logp = capture.capture_sequence(model, ids, 2, "abc")
    assert seq["gated_mlp"].shape == (4, 2, 24)
    assert torch.allclose(seq["res_mlp"][-1].float(), torch.as_tensor(acts["res_mlp"]), atol=1e-5)
    assert capture.hook_inventory(model) == {}


def test_write_scores_and_positive_selection():
    delta = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    direction = np.eye(2)[[0, 1]]                       # [layer, out]
    weight = np.ones((2, 3))
    scores = np.stack([directions.write_scores(delta[l], direction[l], weight) for l in range(2)])
    assert scores.shape == (2, 3)
    chosen = directions.positive_global_selection(scores, 10)
    assert chosen == {0: [0, 2], 1: [1]}               # negatives never selected
    head = directions.write_scores(np.ones(8), np.ones(2), np.ones((2, 8)), num_heads=2)
    assert head.shape == (2,) and np.allclose(head, 8.0)


def test_random_control_disjoint_and_norm_matched():
    target = {"mlp": {0: [1, 2, 3], 1: [0]}, "attention": {1: [0, 1]}}
    control = hooks.random_selection(target, seed=3, widths={"mlp": 6, "attention": 4}, disjoint=True)
    for kind in target:
        for layer, units in target[kind].items():
            assert not set(units) & set(control[kind].get(layer, []))
            assert len(control[kind].get(layer, [])) == len(units)
    raw = {"mlp": np.random.default_rng(0).normal(size=(2, 6)), "attention": np.random.default_rng(1).normal(size=(2, 16))}
    matched, report = hooks.match_selected_norms(raw, raw, control, head_dim=4, target_selection=target)
    for row in report:
        kind, layer = row["kind"], row["layer"]
        idx = steering.channels(kind, control[kind][layer], 4)
        assert np.isclose(np.linalg.norm(matched[kind][layer, idx]), row["l2"])


def test_allocation_conserves_budget():
    quota = importance.allocate(np.array([0.5, -1.0, 2.0, 0.0]), 20, 2, 100)
    assert quota.sum() == 20 and (quota >= 2).all()
    circuit = importance.allocate_circuit(np.ones((2, 2)), np.random.default_rng(0).normal(size=(2, 24)),
                                          np.random.default_rng(0).normal(size=(2, 4)),
                                          budgets={"mlp": 6, "attention": 2}, minimum={"attention": 1, "mlp": 2})
    assert sum(len(v) for v in circuit["mlp"].values()) == 6
    assert sum(len(v) for v in circuit["attention"].values()) == 2


def test_shapley_values_sum_to_full_effect():
    rng = np.random.default_rng(0)
    margins = {a: float(rng.normal()) for a in stage_ablation.ARMS}
    total = margins["a1s1n1"] - margins["a0s0n0"]
    parts = sum(stage_ablation.contrast(margins, stage_ablation.shapley_weights(s)) for s in stage_ablation.STAGES)
    assert np.isclose(parts, total)
    rows = [dict(song_id="x", style="funk", arm=a, margin=m) for a, m in margins.items()]
    summary = stage_ablation.decompose(rows)
    assert np.isclose(sum(summary["funk"]["share"].values()), 1.0)


def test_clap_windows_cover_every_sample():
    frames = 3 * evaluation.WIDTH + 12345
    intervals, weights = evaluation.planned_intervals_and_weights(frames)
    assert np.isclose(weights.sum(), 1.0) and intervals[-1][1] == frames
    out = evaluation.outcome([0.1, 0.5, 0.2, 0.2, 0.1], 1)
    assert out["hit"] and np.isclose(out["target_margin"], 0.3)


def test_causal_probe_zero_budget_reproduces_base():
    model = tiny()
    probe = causal.TeacherForcedProbe(model)
    ids = [2, 4, 7, 5, 9, 11, 3]
    unit = torch.zeros(2, 2, 16)
    unit[:, :, 0] = 1.0
    base, lp = probe.forward(ids, 2, unit, "abc")
    again, lp2 = probe.forward(ids, 2, unit, "abc", edits={"mlp": {}})
    assert torch.equal(base, again) and torch.equal(lp, lp2)
    edits = causal.build_edits(probe, "mlp", {0: [1, 2]}, "zero")
    changed, _ = probe.forward(ids, 2, unit, "abc", edits)
    assert changed.shape == base.shape
    assert capture.hook_inventory(model) == {}


def test_caa_delta_reproduces_residual_vector():
    rng = np.random.default_rng(0)
    down = rng.normal(size=(16, 24))
    weights = {0: (rng.normal(size=(16, 16)), down), 1: (rng.normal(size=(16, 16)), down)}
    v = rng.normal(size=16)
    base = steering.recipe()
    r = steering.caa_recipe(base, v, 1, weights=weights, circuit_norm=2.0)
    written = down @ r["abc_delta"]["mlp"][1]
    assert np.isclose(np.linalg.norm(written), 2.0)
    assert np.isclose(written @ v / np.linalg.norm(written) / np.linalg.norm(v), 1.0)
    assert canonical_selection(r["abc_selection"])["mlp"][1] == list(range(24))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as error:  # noqa: BLE001
                failures += 1
                print("FAIL", name, type(error).__name__, error)
    raise SystemExit(1 if failures else 0)
