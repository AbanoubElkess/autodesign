from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.geometry import sample_layered_material_grid
from src.solver import LumericalFDTDAdapter, SolverSetupError, create_solver
from src.specs import load_problem_spec

from tests.support import write_spec


class SolverAdapterTests(unittest.TestCase):
    def test_lumerical_request_contains_periodic_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_problem_spec(write_spec(Path(tmpdir)))
            solver = create_solver(spec)
            materials = solver.resolve_materials()
            grid = sample_layered_material_grid(spec.geometry, len(materials), generator=torch.Generator().manual_seed(9))
            request = solver.build_request(grid, materials, sample_id="smoke")
            self.assertTrue(request.solver_settings["periodic_x"])
            self.assertTrue(request.solver_settings["periodic_y"])

    def test_lumerical_export_filters_materials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_path = root / "materials_export.json"
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "materials": [
                            {
                                "name": "SiO2",
                                "material_class": "dielectric",
                                "wavelength_range_um": [0.3, 5.0],
                                "n_real": 1.45,
                                "k_imag": 0.0,
                            },
                            {
                                "name": "GST",
                                "material_class": "pcm",
                                "wavelength_range_um": [0.7, 12.0],
                                "n_real": 4.2,
                                "k_imag": 0.15,
                                "phase_change_capable": True,
                            },
                        ]
                    },
                    handle,
                    indent=2,
                )
            spec = load_problem_spec(
                write_spec(
                    root,
                    overrides={
                        "solver": {"backend": "lumerical_fdtd", "material_database_export": str(export_path)},
                        "material_search_policy": {"include_materials": [], "material_classes": ["dielectric", "pcm"]},
                    },
                )
            )
            solver = LumericalFDTDAdapter(spec)
            materials = solver.resolve_materials()
            self.assertEqual([material.name for material in materials], ["GST", "SiO2"])

    def test_lumerical_without_setup_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_problem_spec(
                write_spec(
                    Path(tmpdir),
                    overrides={"solver": {"backend": "lumerical_fdtd", "cli_bridge": []}},
                )
            )
            solver = LumericalFDTDAdapter(spec)
            with self.assertRaises(SolverSetupError):
                solver.resolve_materials()


if __name__ == "__main__":
    unittest.main()
