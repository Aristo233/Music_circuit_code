"""Shared helpers for the command-line scripts: model loading and JSON I/O."""
from __future__ import annotations

import json
from pathlib import Path


def load_pipeline(model_dir, vae_dir, device="cuda:0", memory_budget_gib=28):
    """Native torch-eager, unquantized YuE2 pipeline (hooks need the eager backbone)."""
    import torch
    from yue2 import YuE2Pipeline
    torch.set_num_threads(4)
    return YuE2Pipeline.from_pretrained(str(model_dir), vae=str(vae_dir), device=device,
                                       memory_budget_gib=memory_budget_gib, backend="torch-eager",
                                       quantization="none", local_files_only=True, progress=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def style_texts(dataset):
    """``{style_key: style_text}`` from ``dataset["styles"]``."""
    return {row["style_group"]: row["style_text"] for row in dataset["styles"]}


def split_records(dataset, split):
    return sorted((r for r in dataset["records"] if r["split"] == split), key=lambda r: r["song_id"])
