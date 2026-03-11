"""[PHOT-100] Shared constants for the photonics workflow."""

from __future__ import annotations

from pathlib import Path

SPEC_VERSION = "1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autodesign"
DEFAULT_LUMERICAL_PYTHON_API_PATH = Path(r"C:\api\python")
DEFAULT_LOCAL_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_DATASET_FILE = "dataset.pt"
DEFAULT_MATERIALS_FILE = "materials.json"
DEFAULT_MODEL_FILE = "surrogate.pt"
DEFAULT_CANDIDATES_FILE = "candidates.json"
SUPPORTED_GEOMETRY_FAMILY = "layered_material_grid"
SUPPORTED_SOLVER_BACKENDS = {"mock", "lumerical_fdtd"}
SUPPORTED_POLARIZATIONS = {"TE", "TM"}
SUPPORTED_CHANNELS = {"T", "R", "phase"}
