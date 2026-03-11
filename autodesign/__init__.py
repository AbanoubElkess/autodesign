"""[PHOT-100] Core package for photonics inverse design."""

from .constants import DEFAULT_CACHE_DIR, SPEC_VERSION
from .specs import (
    DatasetConfig,
    DatasetRecord,
    InverseDesignRun,
    LayeredMaterialGrid,
    MaterialSearchPolicy,
    ProblemSpec,
    SimulationResult,
    SolverConfig,
    SolverRequest,
    SpectralGrid,
    SpectralObjective,
    SurrogateConfig,
    load_problem_spec,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "SPEC_VERSION",
    "DatasetConfig",
    "DatasetRecord",
    "InverseDesignRun",
    "LayeredMaterialGrid",
    "MaterialSearchPolicy",
    "ProblemSpec",
    "SimulationResult",
    "SolverConfig",
    "SolverRequest",
    "SpectralGrid",
    "SpectralObjective",
    "SurrogateConfig",
    "load_problem_spec",
]
