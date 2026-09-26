#!/usr/bin/env python3
"""Score-stage sublayer importance for the KL-weighted directions (Appendix figure).

Writes ``importance.npz`` with ``values [lyric, style, site, layer, alpha]``,
the KL-removed fraction and the mean over lyrics and alphas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_pipeline, read_json, style_texts
from music_circuit import importance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True, help="klw_abc.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    texts = style_texts(read_json(args.dataset))
    histories = read_json(args.histories)["records"]
    bank = dict(np.load(args.bank, allow_pickle=False))
    aligned, reference = importance.reference_direction(bank["residual_directions"])
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.model, args.vae, args.device)
    try:
        probe = importance.SublayerProbe(pipe._load_model())
        sigma = importance.sigma_pass(probe, pipe.tokenizer, histories)
        ref = torch.as_tensor(reference, device=probe.device, dtype=torch.float32)
        values, uniform, removed = [], [], []
        for row in histories:
            v, u, r = importance.sublayer_importance(probe, pipe.tokenizer, row["lyrics"], row["abc_ids"], texts, aligned, ref,
                                                     sigma, batch_size=args.batch_size, seed=row.get("seed", 0))
            values.append(v), uniform.append(u), removed.append(r)
            print(row["song_id"], "done", flush=True)
    finally:
        pipe.close()
    values, uniform, removed = (np.stack(x) for x in (values, uniform, removed))
    np.savez_compressed(args.output / "importance.npz", values=values, uniform=uniform, kl_removed=removed, sigma=sigma,
                        alphas=np.asarray(importance.ALPHAS), styles=bank["styles"], mean=values.mean(axis=(0, 4)))
    print("wrote", args.output / "importance.npz")


if __name__ == "__main__":
    main()
