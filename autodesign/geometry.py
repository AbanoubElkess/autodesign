"""[PHOT-101] Geometry encoding utilities for layered material grids."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .specs import LayeredMaterialGrid


def sample_layered_material_grid(
    geometry: LayeredMaterialGrid,
    num_materials: int,
    generator: torch.Generator,
) -> torch.Tensor:
    shape = (geometry.num_layers, geometry.grid_shape[0], geometry.grid_shape[1])
    return torch.randint(0, num_materials, shape, generator=generator, dtype=torch.long)


def serialize_material_grid(material_grid: torch.Tensor) -> list[list[list[int]]]:
    if material_grid.ndim != 3:
        raise ValueError("material_grid must have shape [layers, height, width].")
    return material_grid.detach().cpu().tolist()


def deserialize_material_grid(serialized: list[list[list[int]]]) -> torch.Tensor:
    tensor = torch.tensor(serialized, dtype=torch.long)
    if tensor.ndim != 3:
        raise ValueError("Serialized grid must represent [layers, height, width].")
    return tensor


def one_hot_encode_grid(material_grid: torch.Tensor, num_materials: int) -> torch.Tensor:
    if material_grid.ndim != 3:
        raise ValueError("material_grid must have shape [layers, height, width].")
    return F.one_hot(material_grid.long(), num_classes=num_materials).float()


def soft_assignments_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits / max(temperature, 1e-6), dim=-1)


def hard_assignments_from_soft(assignments: torch.Tensor) -> torch.Tensor:
    if assignments.ndim != 5:
        raise ValueError("assignments must have shape [batch, layers, height, width, materials].")
    return assignments.argmax(dim=-1)


def flatten_assignment_channels(assignments: torch.Tensor) -> torch.Tensor:
    if assignments.ndim != 5:
        raise ValueError("assignments must have shape [batch, layers, height, width, materials].")
    batch, layers, height, width, materials = assignments.shape
    return assignments.permute(0, 1, 4, 2, 3).reshape(batch, layers * materials, height, width)


def geometry_fill_fractions(assignments: torch.Tensor) -> torch.Tensor:
    if assignments.ndim != 5:
        raise ValueError("assignments must have shape [batch, layers, height, width, materials].")
    return assignments.mean(dim=(2, 3))


def geometry_payload(material_grid: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "material_indices": serialize_material_grid(material_grid),
        "metadata": metadata or {},
    }
