"""[PHOT-101] Problem contracts and spec loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CANDIDATES_FILE,
    DEFAULT_DATASET_FILE,
    DEFAULT_LUMERICAL_PYTHON_API_PATH,
    DEFAULT_MATERIALS_FILE,
    DEFAULT_MODEL_FILE,
    SPEC_VERSION,
    SUPPORTED_CHANNELS,
    SUPPORTED_GEOMETRY_FAMILY,
    SUPPORTED_POLARIZATIONS,
    SUPPORTED_SOLVER_BACKENDS,
)


def _to_tuple(values: Any, cast) -> tuple[Any, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected list/tuple, received {type(values)!r}")
    return tuple(cast(value) for value in values)


def _normalize_response_map(target_response: dict[str, Any]) -> dict[str, tuple[tuple[float, ...], ...]]:
    normalized: dict[str, tuple[tuple[float, ...], ...]] = {}
    for channel, rows in target_response.items():
        normalized[channel] = tuple(tuple(float(value) for value in row) for row in rows)
    return normalized


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_dataclass_to_dict(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SpectralGrid:
    wavelength_um: tuple[float, ...]
    incidence_angles_deg: tuple[float, ...] = (0.0,)
    polarizations: tuple[str, ...] = ("TE",)

    def __post_init__(self) -> None:
        if not self.wavelength_um:
            raise ValueError("spectral_grid.wavelength_um must contain at least one wavelength.")
        if not self.incidence_angles_deg:
            raise ValueError("spectral_grid.incidence_angles_deg must contain at least one angle.")
        invalid_polarizations = set(self.polarizations) - SUPPORTED_POLARIZATIONS
        if invalid_polarizations:
            raise ValueError(f"Unsupported polarizations: {sorted(invalid_polarizations)}")

    @property
    def condition_count(self) -> int:
        return len(self.incidence_angles_deg) * len(self.polarizations)

    def condition_labels(self) -> list[str]:
        labels = []
        for angle in self.incidence_angles_deg:
            for polarization in self.polarizations:
                labels.append(f"{polarization}@{angle:.2f}deg")
        return labels


@dataclass(frozen=True)
class SpectralObjective:
    target_channels: tuple[str, ...]
    target_response: dict[str, tuple[tuple[float, ...], ...]]
    channel_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_channels:
            raise ValueError("objective.target_channels must not be empty.")
        invalid_channels = set(self.target_channels) - SUPPORTED_CHANNELS
        if invalid_channels:
            raise ValueError(f"Unsupported target channels: {sorted(invalid_channels)}")
        missing_response = [channel for channel in self.target_channels if channel not in self.target_response]
        if missing_response:
            raise ValueError(f"objective.target_response is missing channels: {missing_response}")
        for channel in self.target_channels:
            self.channel_weights.setdefault(channel, 1.0)

    def validate(self, spectral_grid: SpectralGrid) -> None:
        expected_conditions = spectral_grid.condition_count
        expected_wavelengths = len(spectral_grid.wavelength_um)
        for channel in self.target_channels:
            rows = self.target_response[channel]
            if len(rows) != expected_conditions:
                raise ValueError(
                    f"Channel {channel!r} has {len(rows)} conditions but expected {expected_conditions}."
                )
            for row in rows:
                if len(row) != expected_wavelengths:
                    raise ValueError(
                        f"Channel {channel!r} row length {len(row)} does not match "
                        f"{expected_wavelengths} wavelengths."
                    )


@dataclass(frozen=True)
class LayeredMaterialGrid:
    family: str
    grid_shape: tuple[int, int]
    unit_cell_size_um: tuple[float, float]
    layer_thickness_um: tuple[float, ...]
    periodic_x: bool = True
    periodic_y: bool = True
    superstrate_material: str = "Air"
    substrate_material: str = "SiO2"

    def __post_init__(self) -> None:
        if self.family != SUPPORTED_GEOMETRY_FAMILY:
            raise ValueError(f"Unsupported geometry family {self.family!r}.")
        if len(self.grid_shape) != 2 or min(self.grid_shape) <= 0:
            raise ValueError("geometry.grid_shape must be a two-element positive integer tuple.")
        if len(self.unit_cell_size_um) != 2 or min(self.unit_cell_size_um) <= 0:
            raise ValueError("geometry.unit_cell_size_um must contain positive values.")
        if not self.layer_thickness_um or min(self.layer_thickness_um) <= 0:
            raise ValueError("geometry.layer_thickness_um must contain positive layer thicknesses.")
        if not self.periodic_x or not self.periodic_y:
            raise ValueError("v1 requires periodic_x and periodic_y to both be true.")

    @property
    def num_layers(self) -> int:
        return len(self.layer_thickness_um)


@dataclass(frozen=True)
class MaterialSearchPolicy:
    material_classes: tuple[str, ...] = ("dielectric", "pcm")
    include_materials: tuple[str, ...] = ()
    exclude_materials: tuple[str, ...] = ()
    max_candidates: int = 4
    prefer_phase_change: bool = False
    database: str = "lumerical"

    def __post_init__(self) -> None:
        if self.max_candidates < 2:
            raise ValueError("material_search_policy.max_candidates must be at least 2.")


@dataclass(frozen=True)
class SolverConfig:
    backend: str = "lumerical_fdtd"
    product: str = "fdtd"
    python_api_path: str | None = (
        str(DEFAULT_LUMERICAL_PYTHON_API_PATH) if DEFAULT_LUMERICAL_PYTHON_API_PATH.exists() else None
    )
    lumapi_module: str = "lumapi"
    executable_path: str | None = None
    cli_bridge: tuple[str, ...] = ()
    material_database_export: str | None = None
    project_template: str | None = None
    timeout_s: int = 600

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_SOLVER_BACKENDS:
            raise ValueError(f"Unsupported solver backend {self.backend!r}.")
        if self.product.lower() != "fdtd":
            raise ValueError("v1 only supports Lumerical FDTD-compatible problem specs.")


@dataclass(frozen=True)
class DatasetConfig:
    num_samples: int = 64
    train_split: float = 0.8
    dataset_file: str = DEFAULT_DATASET_FILE

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("dataset.num_samples must be positive.")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("dataset.train_split must be between 0 and 1.")


@dataclass(frozen=True)
class SurrogateConfig:
    hidden_channels: int = 32
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("surrogate parameters must all be positive.")


@dataclass(frozen=True)
class InverseDesignRun:
    steps: int = 64
    learning_rate: float = 0.1
    top_k: int = 3
    binarization_weight: float = 0.01
    restarts: int = 1
    temperature: float = 1.0
    validate_with_solver: bool = False
    append_refined_to_dataset: bool = True

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.top_k <= 0 or self.restarts <= 0:
            raise ValueError("inverse_design steps, top_k, and restarts must be positive.")


@dataclass(frozen=True)
class SolverRequest:
    problem_id: str
    sample_id: str
    geometry_indices: list[list[list[int]]]
    resolved_materials: tuple[str, ...]
    wavelengths_um: tuple[float, ...]
    incidence_angles_deg: tuple[float, ...]
    polarizations: tuple[str, ...]
    target_channels: tuple[str, ...]
    solver_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class SimulationResult:
    problem_id: str
    sample_id: str
    geometry_indices: list[list[list[int]]]
    resolved_materials: tuple[str, ...]
    spectral_outputs: dict[str, tuple[tuple[float, ...], ...]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    geometry_indices: list[list[list[int]]]
    resolved_materials: tuple[str, ...]
    spectral_outputs: dict[str, tuple[tuple[float, ...], ...]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    random_seed: int
    cache_dir: Path
    output_dir: Path
    geometry: LayeredMaterialGrid
    spectral_grid: SpectralGrid
    objective: SpectralObjective
    material_search_policy: MaterialSearchPolicy = field(default_factory=MaterialSearchPolicy)
    solver: SolverConfig = field(default_factory=SolverConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    inverse_design: InverseDesignRun = field(default_factory=InverseDesignRun)
    materials_file: str = DEFAULT_MATERIALS_FILE
    model_file: str = DEFAULT_MODEL_FILE
    candidates_file: str = DEFAULT_CANDIDATES_FILE
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        self.objective.validate(self.spectral_grid)
        if not self.problem_id:
            raise ValueError("problem_id must not be empty.")

    @property
    def problem_dir(self) -> Path:
        return self.cache_dir / self.problem_id

    @property
    def dataset_path(self) -> Path:
        return self.problem_dir / self.dataset.dataset_file

    @property
    def materials_path(self) -> Path:
        return self.problem_dir / self.materials_file

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.problem_id / self.model_file

    @property
    def candidates_path(self) -> Path:
        return self.output_dir / self.problem_id / self.candidates_file

    def ensure_runtime_dirs(self) -> None:
        self.problem_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.problem_id).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def problem_spec_from_dict(data: dict[str, Any], base_dir: Path | None = None) -> ProblemSpec:
    base = base_dir or Path.cwd()
    geometry = LayeredMaterialGrid(
        family=data["geometry"].get("family", SUPPORTED_GEOMETRY_FAMILY),
        grid_shape=_to_tuple(data["geometry"]["grid_shape"], int),
        unit_cell_size_um=_to_tuple(data["geometry"]["unit_cell_size_um"], float),
        layer_thickness_um=_to_tuple(data["geometry"]["layer_thickness_um"], float),
        periodic_x=bool(data["geometry"].get("periodic_x", True)),
        periodic_y=bool(data["geometry"].get("periodic_y", True)),
        superstrate_material=str(data["geometry"].get("superstrate_material", "Air")),
        substrate_material=str(data["geometry"].get("substrate_material", "SiO2")),
    )
    spectral_grid = SpectralGrid(
        wavelength_um=_to_tuple(data["spectral_grid"]["wavelength_um"], float),
        incidence_angles_deg=_to_tuple(data["spectral_grid"].get("incidence_angles_deg", [0.0]), float),
        polarizations=_to_tuple(data["spectral_grid"].get("polarizations", ["TE"]), str),
    )
    objective = SpectralObjective(
        target_channels=_to_tuple(data["objective"]["target_channels"], str),
        target_response=_normalize_response_map(data["objective"]["target_response"]),
        channel_weights={key: float(value) for key, value in data["objective"].get("channel_weights", {}).items()},
    )
    material_policy = MaterialSearchPolicy(
        material_classes=_to_tuple(data.get("material_search_policy", {}).get("material_classes", ["dielectric", "pcm"]), str),
        include_materials=_to_tuple(data.get("material_search_policy", {}).get("include_materials", []), str),
        exclude_materials=_to_tuple(data.get("material_search_policy", {}).get("exclude_materials", []), str),
        max_candidates=int(data.get("material_search_policy", {}).get("max_candidates", 4)),
        prefer_phase_change=bool(data.get("material_search_policy", {}).get("prefer_phase_change", False)),
        database=str(data.get("material_search_policy", {}).get("database", "lumerical")),
    )
    solver = SolverConfig(
        backend=str(data.get("solver", {}).get("backend", "lumerical_fdtd")),
        product=str(data.get("solver", {}).get("product", "fdtd")),
        python_api_path=data.get("solver", {}).get(
            "python_api_path",
            str(DEFAULT_LUMERICAL_PYTHON_API_PATH) if DEFAULT_LUMERICAL_PYTHON_API_PATH.exists() else None,
        ),
        lumapi_module=str(data.get("solver", {}).get("lumapi_module", "lumapi")),
        executable_path=data.get("solver", {}).get("executable_path"),
        cli_bridge=_to_tuple(data.get("solver", {}).get("cli_bridge", []), str),
        material_database_export=data.get("solver", {}).get("material_database_export"),
        project_template=data.get("solver", {}).get("project_template"),
        timeout_s=int(data.get("solver", {}).get("timeout_s", 600)),
    )
    dataset = DatasetConfig(
        num_samples=int(data.get("dataset", {}).get("num_samples", 64)),
        train_split=float(data.get("dataset", {}).get("train_split", 0.8)),
        dataset_file=str(data.get("dataset", {}).get("dataset_file", DEFAULT_DATASET_FILE)),
    )
    surrogate = SurrogateConfig(
        hidden_channels=int(data.get("surrogate", {}).get("hidden_channels", 32)),
        epochs=int(data.get("surrogate", {}).get("epochs", 12)),
        batch_size=int(data.get("surrogate", {}).get("batch_size", 8)),
        learning_rate=float(data.get("surrogate", {}).get("learning_rate", 1e-3)),
        weight_decay=float(data.get("surrogate", {}).get("weight_decay", 1e-4)),
    )
    inverse = InverseDesignRun(
        steps=int(data.get("inverse_design", {}).get("steps", 64)),
        learning_rate=float(data.get("inverse_design", {}).get("learning_rate", 0.1)),
        top_k=int(data.get("inverse_design", {}).get("top_k", 3)),
        binarization_weight=float(data.get("inverse_design", {}).get("binarization_weight", 0.01)),
        restarts=int(data.get("inverse_design", {}).get("restarts", 1)),
        temperature=float(data.get("inverse_design", {}).get("temperature", 1.0)),
        validate_with_solver=bool(data.get("inverse_design", {}).get("validate_with_solver", False)),
        append_refined_to_dataset=bool(data.get("inverse_design", {}).get("append_refined_to_dataset", True)),
    )
    cache_dir = Path(data.get("cache_dir", DEFAULT_CACHE_DIR))
    output_dir = Path(data.get("output_dir", DEFAULT_CACHE_DIR / "runs"))
    if not cache_dir.is_absolute():
        cache_dir = (base / cache_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    return ProblemSpec(
        problem_id=str(data["problem_id"]),
        random_seed=int(data.get("random_seed", 7)),
        cache_dir=cache_dir,
        output_dir=output_dir,
        geometry=geometry,
        spectral_grid=spectral_grid,
        objective=objective,
        material_search_policy=material_policy,
        solver=solver,
        dataset=dataset,
        surrogate=surrogate,
        inverse_design=inverse,
        materials_file=str(data.get("materials_file", DEFAULT_MATERIALS_FILE)),
        model_file=str(data.get("model_file", DEFAULT_MODEL_FILE)),
        candidates_file=str(data.get("candidates_file", DEFAULT_CANDIDATES_FILE)),
        spec_version=str(data.get("spec_version", SPEC_VERSION)),
    )


def load_problem_spec(path: str | Path) -> ProblemSpec:
    spec_path = Path(path).resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return problem_spec_from_dict(data, base_dir=spec_path.parent)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
