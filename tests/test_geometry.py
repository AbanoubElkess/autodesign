from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from autodesign.geometry import (
    deserialize_material_grid,
    one_hot_encode_grid,
    sample_layered_material_grid,
    serialize_material_grid,
)
from autodesign.specs import load_problem_spec

from tests.support import write_spec


class GeometryEncodingTests(unittest.TestCase):
    def test_round_trip_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_problem_spec(write_spec(Path(tmpdir)))
            grid = sample_layered_material_grid(spec.geometry, num_materials=3, generator=torch.Generator().manual_seed(4))
            payload = serialize_material_grid(grid)
            restored = deserialize_material_grid(payload)
            self.assertTrue(torch.equal(grid, restored))

    def test_one_hot_shape_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_problem_spec(write_spec(Path(tmpdir)))
            grid = sample_layered_material_grid(spec.geometry, num_materials=3, generator=torch.Generator().manual_seed(4))
            encoded = one_hot_encode_grid(grid, num_materials=3)
            self.assertEqual(tuple(encoded.shape), (spec.geometry.num_layers, 4, 4, 3))


if __name__ == "__main__":
    unittest.main()
