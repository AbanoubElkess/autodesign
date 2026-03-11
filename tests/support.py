"""Test helpers for the autodesign workflow."""

from __future__ import annotations

import json
import sys
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any


def write_lumerical_cli_bridge(root: Path) -> Path:
    bridge_path = root / "lumerical_cli_bridge.py"
    bridge_path.write_text(
        textwrap.dedent(
            """
            import json
            import math
            import sys


            def flatten(values):
                for item in values:
                    if isinstance(item, list):
                        yield from flatten(item)
                    else:
                        yield item


            def build_response(request):
                geometry = request["geometry_indices"]
                flat = list(flatten(geometry))
                mean_value = sum(flat) / max(len(flat), 1)
                edge_hint = sum(1 for idx in range(1, len(flat)) if flat[idx] != flat[idx - 1]) / max(len(flat), 1)
                outputs = {}
                condition_rows = {channel: [] for channel in request["target_channels"]}
                for angle in request["incidence_angles_deg"]:
                    angle_factor = 1.0 - min(abs(angle) / 90.0, 0.5)
                    for polarization in request["polarizations"]:
                        pol_factor = 1.0 if polarization == "TE" else 0.94
                        transmission = []
                        reflection = []
                        phase = []
                        for wavelength in request["wavelengths_um"]:
                            base = 0.45 + 0.08 * mean_value + 0.12 * math.cos(2 * math.pi * wavelength / 1.55)
                            t = max(0.0, min(base * angle_factor * pol_factor, 0.98))
                            r = max(0.0, min(0.08 + 0.1 * edge_hint + 0.03 * abs(mean_value - 1.5), 0.8))
                            total = max(t + r, 1.0)
                            transmission.append(t / total)
                            reflection.append(r / total)
                            phase.append(math.atan2(math.sin(base * math.pi), math.cos(base * math.pi)))
                        if "T" in condition_rows:
                            condition_rows["T"].append(transmission)
                        if "R" in condition_rows:
                            condition_rows["R"].append(reflection)
                        if "phase" in condition_rows:
                            condition_rows["phase"].append(phase)
                for channel, rows in condition_rows.items():
                    outputs[channel] = rows
                return {
                    "result": {
                        "problem_id": request["problem_id"],
                        "sample_id": request["sample_id"],
                        "geometry_indices": geometry,
                        "resolved_materials": request["resolved_materials"],
                        "spectral_outputs": outputs,
                        "provenance": {"backend": "lumerical_cli_test_bridge"},
                    }
                }


            def main():
                request_path, response_path = sys.argv[1], sys.argv[2]
                with open(request_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload["action"] == "list_materials":
                    response = {
                        "materials": [
                            {
                                "name": "SiO2",
                                "material_class": "dielectric",
                                "wavelength_range_um": [0.3, 5.0],
                                "n_real": 1.45,
                                "k_imag": 0.0,
                            },
                            {
                                "name": "SiN",
                                "material_class": "dielectric",
                                "wavelength_range_um": [0.35, 4.0],
                                "n_real": 2.02,
                                "k_imag": 0.001,
                            },
                            {
                                "name": "GST",
                                "material_class": "pcm",
                                "wavelength_range_um": [0.7, 12.0],
                                "n_real": 4.2,
                                "k_imag": 0.15,
                                "phase_change_capable": True,
                            },
                            {
                                "name": "TiO2",
                                "material_class": "dielectric",
                                "wavelength_range_um": [0.35, 2.5],
                                "n_real": 2.35,
                                "k_imag": 0.002,
                            },
                        ]
                    }
                elif payload["action"] == "simulate":
                    response = build_response(payload["request"])
                else:
                    raise ValueError(f"Unsupported action {payload['action']!r}")
                with open(response_path, "w", encoding="utf-8") as handle:
                    json.dump(response, handle, indent=2)


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    return bridge_path


def build_spec_dict(root: Path) -> dict[str, Any]:
    bridge_path = write_lumerical_cli_bridge(root)
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
            "backend": "lumerical_fdtd",
            "product": "fdtd",
            "cli_bridge": [sys.executable, str(bridge_path)],
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
