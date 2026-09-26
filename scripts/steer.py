#!/usr/bin/env python3
"""Generate songs under the dynamic circuit and the baselines, then score them with CLAP.

Arms: ``baseline`` (empty tag), ``prompt`` (style text in all stages),
``circuit`` (the paper's recipe), ``random`` (disjoint random circuit, norm
matched) and ``caa`` (mean residual difference after one layer). All circuit
arms share the music-token-stage intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_pipeline, read_json, write_json, split_records
from music_circuit import directions, evaluation, steering
from music_circuit.constants import STYLES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--klw-bank", type=Path, required=True, help="klw_abc.npz")
    p.add_argument("--klw-circuits", type=Path, required=True, help="circuits_klw_abc.json")
    p.add_argument("--music-bank", type=Path, required=True, help="mean128_semantic.npz")
    p.add_argument("--music-circuits", type=Path, required=True, help="single-position music-token circuits json")
    p.add_argument("--music-reference", type=Path, required=True, help="single-position direction bank npz (norm reference)")
    p.add_argument("--arms", nargs="+", default=("baseline", "prompt", "circuit"))
    p.add_argument("--styles", nargs="+", default=STYLES)
    p.add_argument("--seeds", type=int, nargs=2, default=(1, 2), metavar=("ABC_SEED", "MUSIC_SEED"))
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--caa-layer", type=int, default=4)
    p.add_argument("--caa-scale", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clap-checkpoint", type=Path, default=None)
    p.add_argument("--model", type=Path, default=Path("models/YuE2-3B"))
    p.add_argument("--vae", type=Path, default=Path("models/YuE2-Vae"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    import soundfile as sf
    dataset = read_json(args.dataset)
    klw = dict(np.load(args.klw_bank, allow_pickle=False))
    klw_circuits = read_json(args.klw_circuits)["abc"]
    music = dict(np.load(args.music_bank, allow_pickle=False))
    music_circuits = read_json(args.music_circuits)["semantic"]
    music_reference = dict(np.load(args.music_reference, allow_pickle=False))
    pipe = load_pipeline(args.model, args.vae, args.device)
    encoder = None
    rows = []
    try:
        weights = directions.projection_weights(pipe._load_model())
        for record in split_records(dataset, args.split):
            for style in args.styles:
                circuit = steering.circuit_recipe(style, klw, klw_circuits, music, music_circuits, music_reference)
                recipes = {"baseline": steering.recipe(), "prompt": steering.prompt_recipe(style), "circuit": circuit}
                if "random" in args.arms:
                    recipes["random"] = steering.random_circuit_recipe(circuit, seed=args.random_seed)
                if "caa" in args.arms:
                    si = list(klw["styles"]).index(style)
                    norm = steering.circuit_residual_write_norm(
                        {k: klw["component_delta_" + k][si] for k in ("mlp", "attention")}, klw_circuits[style], weights)
                    recipes["caa"] = steering.caa_recipe(circuit, music["residual_delta"][si, 1, args.caa_layer] if "residual_delta" in music
                                                         else klw["residual_delta"][si, 1, args.caa_layer],
                                                         args.caa_layer, weights=weights, circuit_norm=norm, scale=args.caa_scale)
                for arm in args.arms:
                    case = f"{record['song_id']}__{style}__{arm}__{args.seeds[0]}_{args.seeds[1]}"
                    folder = args.output / case
                    if (folder / "result.json").exists():
                        rows.append(read_json(folder / "result.json"))
                        continue
                    song = steering.generate_song(pipe, record["lyrics"], recipes[arm], abc_seed=args.seeds[0],
                                                  music_seed=args.seeds[1], song_id=record["song_id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "score.abc").write_text(song["abc"])
                    np.save(folder / "semantic.npy", np.asarray(song["semantic_tokens"], np.int32))
                    sf.write(folder / "audio.flac", song["waveform"], song["sample_rate"], subtype="PCM_24")
                    if encoder is None:
                        encoder = evaluation.LaionClapEncoder(args.clap_checkpoint, device=args.device)
                    clap = evaluation.score_song(encoder, song["waveform"], style, complete=song["complete"])
                    result = dict(case=case, song_id=record["song_id"], style=style, arm=arm, complete=song["complete"],
                                  abc_tokens=len(song["abc_ids"]), music_tokens=len(song["semantic_tokens"]),
                                  audio_seconds=len(song["waveform"]) / song["sample_rate"], clap=clap, audits=song["audits"])
                    write_json(folder / "result.json", result)
                    rows.append(result)
                    print(json.dumps(dict(case=case, hit=clap["hit"], margin=round(clap["target_margin"], 3))), flush=True)
    finally:
        pipe.close()
    summary = {}
    for arm in args.arms:
        sel = [r for r in rows if r["arm"] == arm]
        summary[arm] = dict(cases=len(sel), hits=sum(r["clap"]["hit"] for r in sel),
                            mean_margin=float(np.mean([r["clap"]["target_margin"] for r in sel])) if sel else None,
                            complete=sum(r["complete"] for r in sel))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
