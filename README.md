# autodesign

`autodesign` is a photonics inverse-design toolkit built from this repository's lightweight script-first architecture. The repository now targets periodic multi-material metasurface unit cells as the validated v1 slice, while keeping the internal contracts general enough to extend toward broader photonics structures in later phases.

## Current Scope

- `[PHOT-100]` Bootstrap the repository around `autodesign` and move runtime caches to `~/.cache/autodesign` by default.
- `[PHOT-101]` Define a JSON problem contract for layered multi-material periodic unit cells and spectral-response objectives.
- `[PHOT-102]` Use Lumerical FDTD as the default solver interface for dataset generation, validation, and refinement.
- `[PHOT-103]` Build cached datasets from sampled geometries and solver responses.
- `[PHOT-104]` Train a torch-only forward surrogate on geometry-to-spectrum data.
- `[PHOT-105]` Run surrogate-first inverse design, emit candidate layouts, and refine them back through the solver.
- `[PHOT-106]` Guard the workflow with automated tests and opt-in solver integration points.

v1 does not claim arbitrary photonics support in execution. The stable abstraction layer is general; the shipped implementation is intentionally narrow enough to stay testable.

## Repository Layout

- `prepare.py`: thin CLI for `materials` and `dataset` preparation stages.
- `train.py`: thin CLI for `surrogate`, `inverse`, and `refine` runtime modes.
- `autodesign/`: internal package for specs, geometry encodings, materials, solver adapters, datasets, surrogate modeling, and workflow orchestration.
- `examples/mock_periodic_unit_cell.json`: periodic unit-cell reference problem configured for Lumerical FDTD.
- `program.md`: optional automation guidance, not a required runtime dependency.
- `swarm.py`: thin CLI for a local-only Ollama swarm that iterates on the inverse-design loop.
- `tests/`: unit and workflow coverage for the v1 slice.

## Problem Spec

The public contract is a JSON file passed to both root entrypoints:

```bash
python prepare.py --spec examples/mock_periodic_unit_cell.json --stage dataset
python train.py --spec examples/mock_periodic_unit_cell.json --mode surrogate
python train.py --spec examples/mock_periodic_unit_cell.json --mode inverse
python train.py --spec examples/mock_periodic_unit_cell.json --mode refine
```

For a local-only swarm run:

```bash
python swarm.py --spec examples/freeform_metasurface_swarm.json --model gemma3:4b --rounds 1
```

The spec defines:

- geometry: layered 2.5D material grid, unit-cell size, periodicity, layer thicknesses, substrate/superstrate.
- spectral grid: wavelength samples, incidence angles, and polarizations.
- objective: target channels and their desired spectral responses.
- material search policy: candidate classes, explicit includes/excludes, and candidate-count cap.
- solver: `lumerical_fdtd` as the default and expected execution backend.
- dataset, surrogate, inverse design: runtime budgets and artifact filenames.

## Local Ollama Swarm

The swarm layer is local-only by design:

- every agent call goes through the local `ollama` runtime.
- cloud-backed Ollama models are rejected.
- Hugging Face access is not used anywhere in the swarm flow.
- the swarm tunes the search and training knobs around the inverse-design loop; it does not redefine the physical target.

The default agent roles are:

- `photonics_strategist`: adjusts design-space breadth and sampling.
- `surrogate_tuner`: adjusts forward-model capacity and training settings.
- `inverse_tuner`: adjusts inverse-search hyperparameters.
- `reviewer`: merges the strongest proposals into one safe patch for the next round.

## Solver Strategy

The solver path is Lumerical-first:

- `lumerical_fdtd`: adapter that normalizes solver requests and supports:
  - `solver.material_database_export` for resolving materials from an exported Lumerical database snapshot.
  - `solver.cli_bridge` for external-script or executable-backed simulation/refinement.
  - `solver.python_api_path`, which defaults to `C:\api\python` in this repo.

The direct `lumapi` path is intentionally conservative in this build. It is recognized by the adapter, but executable-backed bridges or exported material snapshots are the recommended integration path until a project-specific direct Lumerical bridge is added in the private fork.

## Current Design Target

The current inverse-design target is a periodic freeform metasurface unit cell, not a full finite metasurface aperture.

- `examples/freeform_metasurface_swarm.json` defines a 2-layer 2.5D freeform meta-atom.
- The unit cell is `0.9 um x 0.9 um` with a `6 x 6` freeform material grid in each layer.
- The stack uses an `Air` superstrate and `SiO2` substrate.
- The candidate materials are `SiO2`, `SiN`, `TiO2`, and `GST`.
- The objective is transmission magnitude plus transmission phase shaping over `1.50-1.60 um` for `TE` and `TM` at normal incidence.

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
- Lumerical-style dataset generation, surrogate training, inverse design, and refinement orchestration.
- scripted local-swarm orchestration without live LLM dependencies in tests.
