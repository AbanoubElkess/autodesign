"""Test helpers for the autodesign workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def build_spec_dict(root: Path) -> dict[str, Any]:
    return {
        "problem_id": "test_problem",
        "random_seed": 3,
        "cache_dir": str(root / "cache"),
        "output_dir": str(root / "runs"),
        "geometry": {
            "family": "layered_material_grid",
            "grid_shape": [4, 4],
            "unit_cell_size_um": [0.9, 0.9],
            "layer_thickness_um": [0.25, 0.12],
            "periodic_x": True,
            "periodic_y": True,
            "superstrate_material": "Air",
            "substrate_material": "SiO2",
        },
        "material_search_policy": {
            "material_classes": ["dielectric", "pcm"],
            "include_materials": ["SiO2", "SiN", "GST"],
            "exclude_materials": [],
            "max_candidates": 3,
            "prefer_phase_change": True,
            "database": "lumerical",
        },
        "spectral_grid": {
            "wavelength_um": [1.45, 1.55, 1.65],
            "incidence_angles_deg": [0.0],
            "polarizations": ["TE", "TM"],
        },
        "objective": {
            "target_channels": ["T", "phase"],
            "target_response": {
                "T": [
                    [0.75, 0.8, 0.85],
                    [0.72, 0.77, 0.83],
                ],
                "phase": [
                    [0.0, 0.4, 0.8],
                    [0.0, 0.35, 0.75],
                ],
            },
            "channel_weights": {"T": 1.0, "phase": 0.2},
        },
        "dataset": {
            "num_samples": 8,
            "train_split": 0.75,
            "dataset_file": "dataset.pt",
        },
        "surrogate": {
            "hidden_channels": 8,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "inverse_design": {
            "steps": 6,
            "learning_rate": 0.08,
            "top_k": 2,
            "binarization_weight": 0.01,
            "restarts": 1,
            "temperature": 1.0,
            "validate_with_solver": False,
            "append_refined_to_dataset": True,
        },
        "solver": {
            "backend": "mock",
            "product": "fdtd",
            "timeout_s": 30,
        },
    }


def write_spec(root: Path, overrides: dict[str, Any] | None = None) -> Path:
    spec = build_spec_dict(root)
    if overrides:
        spec = merge_dicts(spec, overrides)
    path = root / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    return path


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
