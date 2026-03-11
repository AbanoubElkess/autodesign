# autodesign program

This file is optional guidance for future automation. It is no longer the core product surface of the repository.

## Mission

Operate on the `[PHOT-*]` roadmap without widening scope beyond the validated v1 slice:

1. `[PHOT-100]` Keep the repository centered on the `autodesign` package with thin root scripts.
2. `[PHOT-101]` Preserve the JSON problem contract as the single source of truth.
3. `[PHOT-102]` Treat Lumerical FDTD as the golden solver, but keep the mock backend healthy so automated tests stay runnable.
4. `[PHOT-103]` Store datasets and large runtime artifacts outside git.
5. `[PHOT-104]` Keep surrogate modeling torch-only.
6. `[PHOT-105]` Use surrogate-first inverse design and solver-backed refinement; do not jump straight to unrestricted solver-in-loop topology search.
7. `[PHOT-106]` Extend coverage before broadening the supported structure family.
8. `[PHOT-107]` Keep swarm orchestration local-only through Ollama and reject cloud-backed models.

## Guardrails

- Do not replace the thin `prepare.py` / `train.py` entrypoint model with a sprawling CLI surface unless the JSON contract can no longer support the required workflows.
- Do not claim general support for arbitrary photonics structures until a new structure family has tests, a stable geometry contract, and a validated solver path.
- Do not add solver-specific Python dependencies to the core project. Use bridges, optional integration points, or private-fork specializations instead.
- Do not store generated datasets, checkpoints, or solver outputs in git.
- Do not let the swarm call Hugging Face or any remote model endpoint.
- Keep issue references explicit in work summaries and commits, for example `[PHOT-104] train surrogate on updated dataset schema`.

## Default Execution Order

For a new problem spec:

1. Resolve materials or generate a dataset with `prepare.py`.
2. Train the surrogate with `train.py --mode surrogate`.
3. Generate inverse candidates with `train.py --mode inverse`.
4. Validate and append solver-backed refinements with `train.py --mode refine`.
5. For swarm-guided tuning, run `swarm.py` against a local Ollama model after the baseline problem spec is stable.

## Preferred Extensions

- Add richer Lumerical bridges in the private fork before attempting direct lumapi-heavy integrations in this public code line.
- Add a new geometry family only after it has its own spec validation, dataset sampling rules, and solver request tests.
- Add new optimization regularizers only if they preserve the clean surrogate/inverse/refine separation.
