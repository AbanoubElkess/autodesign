"""[PHOT-103] Dataset persistence and cache pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .geometry import deserialize_material_grid, sample_layered_material_grid
from .materials import MaterialRecord, material_records_to_payload
from .solver import BaseSolverAdapter
from .specs import DatasetRecord, ProblemSpec, SimulationResult, save_json


def simulation_result_to_tensor(result: SimulationResult, spec: ProblemSpec) -> torch.Tensor:
    channels = []
    for channel in spec.objective.target_channels:
        rows = torch.tensor(result.spectral_outputs[channel], dtype=torch.float32)
        channels.append(rows)
    return torch.stack(channels, dim=1)


def save_material_catalog(spec: ProblemSpec, materials: list[MaterialRecord]) -> Path:
    spec.ensure_runtime_dirs()
    payload = {
        "problem_id": spec.problem_id,
        "materials": material_records_to_payload(materials),
    }
    save_json(spec.materials_path, payload)
    return spec.materials_path


def load_material_catalog(path: str | Path) -> list[MaterialRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    from .materials import material_records_from_payload

    return material_records_from_payload(payload["materials"], source="cache")


def generate_dataset(spec: ProblemSpec, solver: BaseSolverAdapter) -> dict[str, Any]:
    spec.ensure_runtime_dirs()
    materials = solver.resolve_materials()
    save_material_catalog(spec, materials)
    generator = torch.Generator().manual_seed(spec.random_seed)
    geometries = []
    responses = []
    records: list[DatasetRecord] = []

    for index in range(spec.dataset.num_samples):
        geometry = sample_layered_material_grid(spec.geometry, len(materials), generator)
        sample_id = f"sample-{index:04d}"
        request = solver.build_request(geometry, materials, sample_id)
        result = solver.simulate(request)
        geometries.append(geometry)
        responses.append(simulation_result_to_tensor(result, spec))
        records.append(
            DatasetRecord(
                sample_id=sample_id,
                geometry_indices=result.geometry_indices,
                resolved_materials=result.resolved_materials,
                spectral_outputs=result.spectral_outputs,
                provenance=result.provenance,
            )
        )

    payload = {
        "problem_id": spec.problem_id,
        "spec_version": spec.spec_version,
        "material_names": [record.name for record in materials],
        "channel_order": list(spec.objective.target_channels),
        "condition_labels": spec.spectral_grid.condition_labels(),
        "wavelength_um": list(spec.spectral_grid.wavelength_um),
        "geometries": torch.stack(geometries).cpu(),
        "responses": torch.stack(responses).cpu(),
        "records": [record.to_dict() for record in records],
    }
    torch.save(payload, spec.dataset_path)
    return payload


def load_dataset(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def append_simulation_results(
    dataset_path: str | Path,
    results: list[SimulationResult],
    spec: ProblemSpec,
) -> dict[str, Any]:
    payload = load_dataset(dataset_path)
    new_geometries = [deserialize_material_grid(result.geometry_indices) for result in results]
    new_responses = [simulation_result_to_tensor(result, spec) for result in results]
    payload["geometries"] = torch.cat([payload["geometries"], torch.stack(new_geometries).cpu()], dim=0)
    payload["responses"] = torch.cat([payload["responses"], torch.stack(new_responses).cpu()], dim=0)
    payload["records"].extend(
        DatasetRecord(
            sample_id=result.sample_id,
            geometry_indices=result.geometry_indices,
            resolved_materials=result.resolved_materials,
            spectral_outputs=result.spectral_outputs,
            provenance=result.provenance,
        ).to_dict()
        for result in results
    )
    torch.save(payload, Path(dataset_path))
    return payload
