"""[PHOT-102] Material filtering and candidate selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .specs import MaterialSearchPolicy


@dataclass(frozen=True)
class MaterialRecord:
    name: str
    material_class: str
    wavelength_range_um: tuple[float, float]
    n_real: float
    k_imag: float
    phase_change_capable: bool = False
    source: str = "builtin"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["wavelength_range_um"] = list(self.wavelength_range_um)
        return payload


BUILTIN_MATERIAL_CATALOG: tuple[MaterialRecord, ...] = (
    MaterialRecord("Air", "superstrate", (0.2, 20.0), 1.0, 0.0),
    MaterialRecord("SiO2", "dielectric", (0.25, 4.5), 1.45, 0.0),
    MaterialRecord("SiN", "dielectric", (0.35, 4.0), 2.02, 0.001),
    MaterialRecord("TiO2", "dielectric", (0.35, 2.5), 2.35, 0.002),
    MaterialRecord("Al2O3", "dielectric", (0.2, 4.5), 1.76, 0.0),
    MaterialRecord("Si", "dielectric", (1.1, 8.0), 3.48, 0.01),
    MaterialRecord("ITO", "conductive_oxide", (1.0, 6.0), 1.9, 0.08),
    MaterialRecord("GST", "pcm", (0.7, 12.0), 4.2, 0.15, phase_change_capable=True),
    MaterialRecord("VO2", "pcm", (0.6, 8.0), 2.7, 0.2, phase_change_capable=True),
    MaterialRecord("Au", "metal", (0.4, 12.0), 0.32, 6.0),
    MaterialRecord("Ag", "metal", (0.3, 12.0), 0.14, 4.3),
)


def material_records_from_payload(payload: list[dict[str, object]], source: str) -> list[MaterialRecord]:
    records = []
    for row in payload:
        records.append(
            MaterialRecord(
                name=str(row["name"]),
                material_class=str(row["material_class"]),
                wavelength_range_um=tuple(float(value) for value in row["wavelength_range_um"]),
                n_real=float(row["n_real"]),
                k_imag=float(row["k_imag"]),
                phase_change_capable=bool(row.get("phase_change_capable", False)),
                source=source,
            )
        )
    return records


def material_records_to_payload(records: list[MaterialRecord]) -> list[dict[str, object]]:
    return [record.to_dict() for record in records]


def filter_material_records(
    records: list[MaterialRecord] | tuple[MaterialRecord, ...],
    policy: MaterialSearchPolicy,
    wavelength_range_um: tuple[float, float],
) -> list[MaterialRecord]:
    lower, upper = wavelength_range_um
    include = set(policy.include_materials)
    exclude = set(policy.exclude_materials)
    classes = set(policy.material_classes)

    candidates = []
    for record in records:
        if record.name in exclude:
            continue
        if include and record.name not in include:
            continue
        if not include and classes and record.material_class not in classes:
            continue
        if record.wavelength_range_um[0] > lower or record.wavelength_range_um[1] < upper:
            continue
        candidates.append(record)

    if not include:
        candidates.sort(
            key=lambda record: (
                0 if policy.prefer_phase_change and record.phase_change_capable else 1,
                record.k_imag,
                -record.n_real,
                record.name,
            )
        )
    else:
        candidates.sort(key=lambda record: (policy.include_materials.index(record.name), record.name))

    return candidates[: policy.max_candidates]


def material_feature_tensor(records: list[MaterialRecord], device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(
        [[record.n_real, record.k_imag] for record in records],
        dtype=torch.float32,
        device=device,
    )
