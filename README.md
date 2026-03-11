# autodesign

`autodesign` is a photonics inverse-design toolkit built from this repository's lightweight script-first architecture. The repository now targets periodic multi-material metasurface unit cells as the validated v1 slice, while keeping the internal contracts general enough to extend toward broader photonics structures in later phases.

## Current Scope

- `[PHOT-100]` Bootstrap the repository around `autodesign` and move runtime caches to `~/.cache/autodesign` by default.
- `[PHOT-101]` Define a JSON problem contract for layered multi-material periodic unit cells and spectral-response objectives.
- `[PHOT-102]` Provide a solver abstraction with a mock backend for local development and a Lumerical FDTD adapter for golden-solver integration.
- `[PHOT-103]` Build cached datasets from sampled geometries and solver responses.
- `[PHOT-104]` Train a torch-only forward surrogate on geometry-to-spectrum data.
- `[PHOT-105]` Run surrogate-first inverse design, emit candidate layouts, and refine them back through the solver.
- `[PHOT-106]` Guard the workflow with automated tests and opt-in solver integration points.

v1 does not claim arbitrary photonics support in execution. The stable abstraction layer is general; the shipped implementation is intentionally narrow enough to stay testable.

## Repository Layout

- `prepare.py`: thin CLI for `materials` and `dataset` preparation stages.
- `train.py`: thin CLI for `surrogate`, `inverse`, and `refine` runtime modes.
- `autodesign/`: internal package for specs, geometry encodings, materials, solver adapters, datasets, surrogate modeling, and workflow orchestration.
- `examples/mock_periodic_unit_cell.json`: runnable mock-backed reference problem.
- `program.md`: optional automation guidance, not a required runtime dependency.
- `tests/`: unit and workflow coverage for the v1 slice.

## Problem Spec

The public contract is a JSON file passed to both root entrypoints:

```bash
python prepare.py --spec examples/mock_periodic_unit_cell.json --stage dataset
python train.py --spec examples/mock_periodic_unit_cell.json --mode surrogate
python train.py --spec examples/mock_periodic_unit_cell.json --mode inverse
python train.py --spec examples/mock_periodic_unit_cell.json --mode refine
```

The spec defines:

- geometry: layered 2.5D material grid, unit-cell size, periodicity, layer thicknesses, substrate/superstrate.
- spectral grid: wavelength samples, incidence angles, and polarizations.
- objective: target channels and their desired spectral responses.
- material search policy: candidate classes, explicit includes/excludes, and candidate-count cap.
- solver: `mock` for local testing or `lumerical_fdtd` for golden-solver integration.
- dataset, surrogate, inverse design: runtime budgets and artifact filenames.

## Solver Strategy

Two solver paths exist:

- `mock`: deterministic local backend for tests and smoke runs.
- `lumerical_fdtd`: adapter that normalizes solver requests and supports:
  - `solver.material_database_export` for resolving materials from an exported Lumerical database snapshot.
  - `solver.cli_bridge` for external-script or executable-backed simulation/refinement.

The direct `lumapi` path is intentionally conservative in this build. It is recognized by the adapter, but executable-backed bridges or exported material snapshots are the recommended integration path until a project-specific direct Lumerical bridge is added in the private fork.

## Artifacts

Per-problem artifacts are written under the configured cache and output roots:

- cache: dataset tensor file and resolved material catalog.
- runs: surrogate checkpoint and inverse-design candidate file.

The dataset stores geometry tensors, response tensors, channel ordering, wavelength grid, and per-sample provenance.

## Testing

Run the automated suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests cover:

- spec validation for the periodic v1 slice.
- geometry encoding round-trips.
- material filtering and solver request generation.
- Lumerical setup failure handling.
- mock-backed dataset generation, surrogate training, inverse design, and refinement.
