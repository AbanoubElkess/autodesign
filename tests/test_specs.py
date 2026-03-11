from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.specs import load_problem_spec

from tests.support import write_spec


class ProblemSpecTests(unittest.TestCase):
    def test_rejects_non_periodic_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = write_spec(Path(tmpdir), overrides={"geometry": {"periodic_x": False}})
            with self.assertRaises(ValueError):
                load_problem_spec(spec_path)

    def test_rejects_missing_target_response_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = write_spec(
                Path(tmpdir),
                overrides={"objective": {"target_response": {"phase": []}}},
            )
            with self.assertRaises(ValueError):
                load_problem_spec(spec_path)

    def test_loads_valid_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = write_spec(Path(tmpdir))
            spec = load_problem_spec(spec_path)
            self.assertEqual(spec.problem_id, "test_problem")
            self.assertEqual(spec.geometry.grid_shape, (4, 4))
            self.assertEqual(spec.spectral_grid.condition_count, 2)


if __name__ == "__main__":
    unittest.main()
