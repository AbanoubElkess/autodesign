"""[PHOT-102] Solver adapters and request normalization."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .geometry import deserialize_material_grid, one_hot_encode_grid, serialize_material_grid
from .materials import (
    BUILTIN_MATERIAL_CATALOG,
    MaterialRecord,
    filter_material_records,
    material_feature_tensor,
    material_records_from_payload,
)
from .specs import ProblemSpec, SimulationResult, SolverRequest, save_json


class SolverSetupError(RuntimeError):
    """Raised when a requested solver backend is not configured correctly."""


class BaseSolverAdapter:
    backend_name = "base"

    def __init__(self, spec: ProblemSpec) -> None:
        self.spec = spec

    def resolve_materials(self) -> list[MaterialRecord]:
        raise NotImplementedError

    def build_request(
        self,
        geometry_indices: torch.Tensor,
        materials: list[MaterialRecord],
        sample_id: str,
    ) -> SolverRequest:
        return SolverRequest(
            problem_id=self.spec.problem_id,
            sample_id=sample_id,
            geometry_indices=serialize_material_grid(geometry_indices),
            resolved_materials=tuple(record.name for record in materials),
            wavelengths_um=self.spec.spectral_grid.wavelength_um,
            incidence_angles_deg=self.spec.spectral_grid.incidence_angles_deg,
            polarizations=self.spec.spectral_grid.polarizations,
            target_channels=self.spec.objective.target_channels,
            solver_settings={
                "backend": self.backend_name,
                "product": self.spec.solver.product,
                "periodic_x": self.spec.geometry.periodic_x,
                "periodic_y": self.spec.geometry.periodic_y,
                "unit_cell_size_um": list(self.spec.geometry.unit_cell_size_um),
                "layer_thickness_um": list(self.spec.geometry.layer_thickness_um),
                "superstrate_material": self.spec.geometry.superstrate_material,
                "substrate_material": self.spec.geometry.substrate_material,
            },
        )

    def simulate(self, request: SolverRequest) -> SimulationResult:
        raise NotImplementedError


class MockSolverAdapter(BaseSolverAdapter):
    backend_name = "mock"

    def resolve_materials(self) -> list[MaterialRecord]:
        wavelength_range = (
            min(self.spec.spectral_grid.wavelength_um),
            max(self.spec.spectral_grid.wavelength_um),
        )
        materials = filter_material_records(BUILTIN_MATERIAL_CATALOG, self.spec.material_search_policy, wavelength_range)
        if not materials:
            raise SolverSetupError("No materials matched the requested policy in the mock catalog.")
        return materials

    def simulate(self, request: SolverRequest) -> SimulationResult:
        material_lookup = {record.name: record for record in BUILTIN_MATERIAL_CATALOG}
        selected_materials = [material_lookup[name] for name in request.resolved_materials]
        material_grid = deserialize_material_grid(request.geometry_indices)
        one_hot = one_hot_encode_grid(material_grid, len(selected_materials))
        fractions = one_hot.mean(dim=(1, 2))
        material_features = material_feature_tensor(selected_materials)
        effective_index = torch.matmul(fractions, material_features[:, 0])
        effective_loss = torch.matmul(fractions, material_features[:, 1])
        horizontal_edges = (material_grid[:, :, 1:] != material_grid[:, :, :-1]).float().mean().item()
        vertical_edges = (material_grid[:, 1:, :] != material_grid[:, :-1, :]).float().mean().item()
        edge_density = 0.5 * (horizontal_edges + vertical_edges)
        thickness = torch.tensor(self.spec.geometry.layer_thickness_um, dtype=torch.float32)
        outputs: dict[str, tuple[tuple[float, ...], ...]] = {}
        condition_values: dict[str, list[tuple[float, ...]]] = {channel: [] for channel in request.target_channels}

        for angle in request.incidence_angles_deg:
            angle_rad = math.radians(angle)
            for polarization in request.polarizations:
                pol_factor = 1.0 if polarization == "TE" else 0.92
                t_values: list[float] = []
                r_values: list[float] = []
                phase_values: list[float] = []
                for wavelength in request.wavelengths_um:
                    optical_path = float(torch.sum(thickness * effective_index).item()) / wavelength
                    attenuation = float(torch.sum(thickness * effective_loss).item()) / wavelength
                    oscillation = math.cos(2.0 * math.pi * optical_path * (1.0 + 0.1 * math.sin(angle_rad)))
                    sigmoid_t = 1.0 / (1.0 + math.exp(-(2.8 - 1.6 * attenuation - 0.7 * edge_density + 0.25 * oscillation)))
                    sigmoid_r = 1.0 / (1.0 + math.exp(-(-0.9 + 1.1 * edge_density + 0.2 * abs(effective_index.mean().item() - 1.8))))
                    transmission = max(0.0, min(sigmoid_t * pol_factor * (1.0 - 0.03 * abs(angle_rad)), 1.0))
                    reflection = max(0.0, min(sigmoid_r * (1.0 + 0.02 * abs(angle_rad)), 1.0))
                    total = max(transmission + reflection, 1.0)
                    transmission /= total
                    reflection /= total
                    phase = math.atan2(
                        math.sin(2.0 * math.pi * optical_path * pol_factor),
                        math.cos(2.0 * math.pi * optical_path * pol_factor),
                    )
                    t_values.append(transmission)
                    r_values.append(reflection)
                    phase_values.append(phase)
                if "T" in condition_values:
                    condition_values["T"].append(tuple(t_values))
                if "R" in condition_values:
                    condition_values["R"].append(tuple(r_values))
                if "phase" in condition_values:
                    condition_values["phase"].append(tuple(phase_values))

        for channel, rows in condition_values.items():
            outputs[channel] = tuple(rows)
        return SimulationResult(
            problem_id=request.problem_id,
            sample_id=request.sample_id,
            geometry_indices=request.geometry_indices,
            resolved_materials=request.resolved_materials,
            spectral_outputs=outputs,
            provenance={"backend": self.backend_name, "edge_density": edge_density},
        )


class LumericalFDTDAdapter(BaseSolverAdapter):
    backend_name = "lumerical_fdtd"

    def _load_lumapi(self):
        if self.spec.solver.python_api_path and self.spec.solver.python_api_path not in sys.path:
            sys.path.insert(0, self.spec.solver.python_api_path)
        try:
            return importlib.import_module(self.spec.solver.lumapi_module)
        except ImportError:
            return None

    def _load_material_database_export(self) -> list[MaterialRecord] | None:
        export_path = self.spec.solver.material_database_export
        if not export_path:
            return None
        path = Path(export_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return material_records_from_payload(payload["materials"], source="lumerical_export")

    def _dispatch_cli(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.spec.solver.cli_bridge:
            raise SolverSetupError(
                "solver.cli_bridge must be configured for CLI-backed Lumerical execution."
            )
        workdir = self.spec.problem_dir / "solver_bridge"
        workdir.mkdir(parents=True, exist_ok=True)
        request_path = workdir / f"{payload['action']}_request.json"
        response_path = workdir / f"{payload['action']}_response.json"
        save_json(request_path, payload)
        command = list(self.spec.solver.cli_bridge) + [str(request_path), str(response_path)]
        subprocess.run(
            command,
            cwd=self.spec.problem_dir,
            check=True,
            timeout=self.spec.solver.timeout_s,
        )
        with response_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def resolve_materials(self) -> list[MaterialRecord]:
        wavelength_range = (
            min(self.spec.spectral_grid.wavelength_um),
            max(self.spec.spectral_grid.wavelength_um),
        )
        exported = self._load_material_database_export()
        if exported is not None:
            materials = filter_material_records(exported, self.spec.material_search_policy, wavelength_range)
            if not materials:
                raise SolverSetupError("Lumerical material export did not contain compatible materials.")
            return materials
        if self.spec.solver.cli_bridge:
            response = self._dispatch_cli({"action": "list_materials"})
            records = material_records_from_payload(response["materials"], source="lumerical_cli")
            materials = filter_material_records(records, self.spec.material_search_policy, wavelength_range)
            if not materials:
                raise SolverSetupError("CLI-backed Lumerical material list did not satisfy the search policy.")
            return materials
        if self._load_lumapi() is not None and self.spec.material_search_policy.include_materials:
            known = {record.name: record for record in BUILTIN_MATERIAL_CATALOG}
            materials = [known[name] for name in self.spec.material_search_policy.include_materials if name in known]
            if materials:
                return materials[: self.spec.material_search_policy.max_candidates]
        raise SolverSetupError(
            "Lumerical FDTD requires one of: solver.material_database_export, solver.cli_bridge, "
            "or explicit material_search_policy.include_materials with a configured lumapi installation."
        )

    def simulate(self, request: SolverRequest) -> SimulationResult:
        if self.spec.solver.cli_bridge:
            response = self._dispatch_cli({"action": "simulate", "request": request.to_dict()})
            result = response["result"]
            return SimulationResult(
                problem_id=result["problem_id"],
                sample_id=result["sample_id"],
                geometry_indices=result["geometry_indices"],
                resolved_materials=tuple(result["resolved_materials"]),
                spectral_outputs={
                    channel: tuple(tuple(float(value) for value in row) for row in rows)
                    for channel, rows in result["spectral_outputs"].items()
                },
                provenance=result.get("provenance", {}),
            )

        lumapi = self._load_lumapi()
        if lumapi is None:
            raise SolverSetupError(
                "Direct Lumerical execution is unavailable. Configure solver.cli_bridge or install lumapi."
            )
        if not self.spec.solver.project_template:
            raise SolverSetupError(
                "solver.project_template is required for direct lumapi simulation in this build."
            )
        raise SolverSetupError(
            "Direct lumapi execution requires a project template bridge. Configure solver.cli_bridge "
            "for executable-backed runs or provide a custom integration in your fork."
        )


def create_solver(spec: ProblemSpec) -> BaseSolverAdapter:
    if spec.solver.backend == "mock":
        return MockSolverAdapter(spec)
    if spec.solver.backend == "lumerical_fdtd":
        return LumericalFDTDAdapter(spec)
    raise SolverSetupError(f"Unknown solver backend {spec.solver.backend!r}.")
