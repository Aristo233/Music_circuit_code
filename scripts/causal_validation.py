#!/usr/bin/env python3
"""Ablation and enhancement curves with the KL-weighted direction as reference.

For every lyric of the histories file and every style, the target circuit at
several budgets is compared with random controls (disjoint from the target) on
the style projection Delta S and on the output KL. Rows are appended to
``rows.jsonl``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_pipeline, read_json, style_texts
from music_circuit import causal, directions

BUDGETS = {"mlp": (0, 28, 56, 112, 224, 392, 784, 1568), "attention": (0, 7, 14, 28, 56, 84, 112, 168)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True, help="klw_abc.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--random-seeds", type=int, nargs="+", default=(1, 2, 3))
    p.add_argument("--overlap", action="store_true", help="allow random controls to overlap the target")
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    texts = style_texts(read_json(args.dataset))
    histories = read_json(args.histories)["records"]
    bank = dict(np.load(args.bank, allow_pickle=False))
    styles = bank["styles"].tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.model, args.vae, args.device)
    try:
        model = pipe._load_model()
        weights = directions.projection_weights(model)
        scores = directions.klw_component_scores(bank, weights)
        probe = causal.TeacherForcedProbe(model)
        with (args.output / "rows.jsonl").open("a") as out:
            for row in histories:
                for si, style in enumerate(styles):
                    for kind in ("mlp", "attention"):
                        selections = {b: directions.positive_global_selection(scores[kind][si], b) for b in BUDGETS[kind]}
                        delta = bank["component_delta_" + kind][si]
                        for result in causal.validate_lyric(probe, pipe.tokenizer, row["lyrics"], row["abc_ids"], texts[style],
                                                            bank["residual_directions"][si], delta, selections, kind=kind,
                                                            alpha=args.alpha, random_seeds=args.random_seeds,
                                                            disjoint=not args.overlap, seed=row.get("seed", 0)):
                            out.write(json.dumps(dict(song_id=row["song_id"], style=style, kind=kind, **result)) + "\n")
                print(row["song_id"], "done", flush=True)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
