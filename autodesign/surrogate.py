"""[PHOT-104][PHOT-105] Surrogate modeling and inverse optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .geometry import flatten_assignment_channels, geometry_fill_fractions, hard_assignments_from_soft
from .specs import ProblemSpec, save_json


class SpectralSurrogate(nn.Module):
    def __init__(
        self,
        num_layers: int,
        grid_shape: tuple[int, int],
        num_materials: int,
        hidden_channels: int,
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        height, width = grid_shape
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        in_channels = num_layers * num_materials
        pooled_dim = hidden_channels * height * width
        self.output_shape = output_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + num_layers * num_materials, hidden_channels * 4),
            nn.GELU(),
            nn.Linear(hidden_channels * 4, output_dim),
        )

    def forward(self, assignments: torch.Tensor) -> torch.Tensor:
        batch = assignments.size(0)
        x = flatten_assignment_channels(assignments)
        encoded = self.encoder(x).reshape(batch, -1)
        fill = geometry_fill_fractions(assignments).reshape(batch, -1)
        logits = self.head(torch.cat([encoded, fill], dim=1))
        return logits.view(batch, *self.output_shape)


@dataclass
class SurrogateArtifacts:
    model: SpectralSurrogate
    response_mean: torch.Tensor
    response_std: torch.Tensor
    material_names: list[str]


def _checkpoint_payload(
    model: SpectralSurrogate,
    response_mean: torch.Tensor,
    response_std: torch.Tensor,
    material_names: list[str],
    metrics: dict[str, float],
    spec: ProblemSpec,
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "response_mean": response_mean.cpu(),
        "response_std": response_std.cpu(),
        "material_names": material_names,
        "metrics": metrics,
        "grid_shape": list(spec.geometry.grid_shape),
        "num_layers": spec.geometry.num_layers,
        "hidden_channels": spec.surrogate.hidden_channels,
        "output_shape": [
            spec.spectral_grid.condition_count,
            len(spec.objective.target_channels),
            len(spec.spectral_grid.wavelength_um),
        ],
    }


def build_model_from_checkpoint(checkpoint: dict[str, Any], spec: ProblemSpec) -> SpectralSurrogate:
    model = SpectralSurrogate(
        num_layers=int(checkpoint["num_layers"]),
        grid_shape=tuple(int(value) for value in checkpoint["grid_shape"]),
        num_materials=len(checkpoint["material_names"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        output_shape=tuple(int(value) for value in checkpoint["output_shape"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_surrogate(spec: ProblemSpec, dataset_payload: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometries = dataset_payload["geometries"].long()
    responses = dataset_payload["responses"].float()
    num_materials = len(dataset_payload["material_names"])
    assignments = F.one_hot(geometries, num_classes=num_materials).float()
    split_index = max(1, int(len(assignments) * spec.dataset.train_split))
    train_dataset = TensorDataset(assignments[:split_index], responses[:split_index])
    val_dataset = TensorDataset(assignments[split_index:], responses[split_index:]) if split_index < len(assignments) else None

    model = SpectralSurrogate(
        num_layers=spec.geometry.num_layers,
        grid_shape=spec.geometry.grid_shape,
        num_materials=num_materials,
        hidden_channels=spec.surrogate.hidden_channels,
        output_shape=(
            spec.spectral_grid.condition_count,
            len(spec.objective.target_channels),
            len(spec.spectral_grid.wavelength_um),
        ),
    ).to(device)

    response_mean = train_dataset.tensors[1].mean(dim=0, keepdim=True)
    response_std = train_dataset.tensors[1].std(dim=0, keepdim=True).clamp_min(1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.surrogate.learning_rate,
        weight_decay=spec.surrogate.weight_decay,
    )
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=spec.surrogate.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=spec.surrogate.batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    last_train_loss = 0.0
    last_val_loss = 0.0
    model.train()
    for _ in range(spec.surrogate.epochs):
        for batch_assignments, batch_targets in train_loader:
            batch_assignments = batch_assignments.to(device)
            batch_targets = ((batch_targets - response_mean) / response_std).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_assignments)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.item())
        if val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch_assignments, batch_targets in val_loader:
                    batch_assignments = batch_assignments.to(device)
                    batch_targets = ((batch_targets - response_mean) / response_std).to(device)
                    predictions = model(batch_assignments)
                    losses.append(float(criterion(predictions, batch_targets).item()))
            last_val_loss = sum(losses) / len(losses)
            model.train()

    spec.ensure_runtime_dirs()
    metrics = {"train_loss": last_train_loss, "val_loss": last_val_loss}
    checkpoint = _checkpoint_payload(
        model=model.cpu(),
        response_mean=response_mean,
        response_std=response_std,
        material_names=list(dataset_payload["material_names"]),
        metrics=metrics,
        spec=spec,
    )
    torch.save(checkpoint, spec.model_path)
    return checkpoint


def load_surrogate_artifacts(spec: ProblemSpec) -> SurrogateArtifacts:
    checkpoint = torch.load(spec.model_path, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, spec)
    model.eval()
    return SurrogateArtifacts(
        model=model,
        response_mean=checkpoint["response_mean"].float(),
        response_std=checkpoint["response_std"].float(),
        material_names=list(checkpoint["material_names"]),
    )


def weighted_objective_loss(spec: ProblemSpec, predictions: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(predictions)
    for channel_index, channel in enumerate(spec.objective.target_channels):
        target[:, :, channel_index, :] = torch.tensor(
            spec.objective.target_response[channel],
            dtype=predictions.dtype,
            device=predictions.device,
        )
    channel_weights = torch.tensor(
        [spec.objective.channel_weights[channel] for channel in spec.objective.target_channels],
        dtype=predictions.dtype,
        device=predictions.device,
    ).view(1, 1, -1, 1)
    return ((predictions - target) ** 2 * channel_weights).mean()


def optimize_inverse_design(spec: ProblemSpec, artifacts: SurrogateArtifacts) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = artifacts.model.to(device)
    response_mean = artifacts.response_mean.to(device)
    response_std = artifacts.response_std.to(device)
    best_candidates: list[dict[str, Any]] = []
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(spec.random_seed)
    layers = spec.geometry.num_layers
    height, width = spec.geometry.grid_shape
    num_materials = len(artifacts.material_names)

    for restart in range(spec.inverse_design.restarts):
        logits = 0.01 * torch.randn(
            1,
            layers,
            height,
            width,
            num_materials,
            generator=generator,
            device=device,
        )
        logits.requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=spec.inverse_design.learning_rate)
        for step in range(spec.inverse_design.steps):
            optimizer.zero_grad(set_to_none=True)
            soft_assignments = torch.softmax(logits / max(spec.inverse_design.temperature, 1e-6), dim=-1)
            normalized_predictions = model(soft_assignments)
            predictions = normalized_predictions * response_std + response_mean
            objective_loss = weighted_objective_loss(spec, predictions)
            binarization = (soft_assignments * (1.0 - soft_assignments)).mean()
            loss = objective_loss + spec.inverse_design.binarization_weight * binarization
            loss.backward()
            optimizer.step()

            hard_assignments = hard_assignments_from_soft(soft_assignments.detach()).cpu()[0]
            candidate = {
                "restart": restart,
                "step": step,
                "loss": float(loss.item()),
                "objective_loss": float(objective_loss.item()),
                "material_indices": hard_assignments.tolist(),
                "predicted_response": predictions.detach().cpu()[0].tolist(),
                "material_names": artifacts.material_names,
            }
            best_candidates.append(candidate)
            best_candidates = sorted(best_candidates, key=lambda item: item["loss"])[: spec.inverse_design.top_k]

    spec.ensure_runtime_dirs()
    payload = {
        "problem_id": spec.problem_id,
        "candidates": best_candidates,
        "channel_order": list(spec.objective.target_channels),
        "wavelength_um": list(spec.spectral_grid.wavelength_um),
        "condition_labels": spec.spectral_grid.condition_labels(),
    }
    save_json(spec.candidates_path, payload)
    return payload
