"""[PHOT-103][PHOT-105] CLI-facing workflow orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import append_simulation_results, generate_dataset, load_dataset, load_material_catalog, save_material_catalog
from .geometry import deserialize_material_grid
from .solver import create_solver
from .specs import ProblemSpec, load_problem_spec
from .surrogate import load_surrogate_artifacts, optimize_inverse_design, train_surrogate


def load_spec(path: str | Path) -> ProblemSpec:
    spec = load_problem_spec(path)
    spec.ensure_runtime_dirs()
    return spec


def run_prepare(spec_path: str | Path, stage: str) -> None:
    spec = load_spec(spec_path)
    solver = create_solver(spec)
    if stage == "materials":
        materials = solver.resolve_materials()
        save_material_catalog(spec, materials)
        print(f"[PHOT-102] Saved {len(materials)} materials to {spec.materials_path}")
        return
    if stage == "dataset":
        payload = generate_dataset(spec, solver)
        print(f"[PHOT-103] Saved dataset with {len(payload['records'])} samples to {spec.dataset_path}")
        return
    raise ValueError(f"Unsupported prepare stage {stage!r}.")


def run_train(spec_path: str | Path, mode: str) -> None:
    spec = load_spec(spec_path)
    if mode == "surrogate":
        dataset_payload = load_dataset(spec.dataset_path)
        checkpoint = train_surrogate(spec, dataset_payload)
        print(
            f"[PHOT-104] Saved surrogate checkpoint to {spec.model_path} "
            f"(train_loss={checkpoint['metrics']['train_loss']:.6f}, "
            f"val_loss={checkpoint['metrics']['val_loss']:.6f})"
        )
        return

    if mode == "inverse":
        artifacts = load_surrogate_artifacts(spec)
        payload = optimize_inverse_design(spec, artifacts)
        print(f"[PHOT-105] Saved {len(payload['candidates'])} inverse candidates to {spec.candidates_path}")
        return

    if mode == "refine":
        solver = create_solver(spec)
        materials = load_material_catalog(spec.materials_path)
        with spec.candidates_path.open("r", encoding="utf-8") as handle:
            candidate_payload = json.load(handle)
        refined_results = []
        for index, candidate in enumerate(candidate_payload["candidates"]):
            geometry = deserialize_material_grid(candidate["material_indices"])
            request = solver.build_request(geometry, materials, sample_id=f"refine-{index:04d}")
            result = solver.simulate(request)
            refined_results.append(result)
        append_simulation_results(spec.dataset_path, refined_results, spec)
        print(
            f"[PHOT-105] Refined {len(refined_results)} candidates and appended them to {spec.dataset_path}"
        )
        return

    raise ValueError(f"Unsupported train mode {mode!r}.")
