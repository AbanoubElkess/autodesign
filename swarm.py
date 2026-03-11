"""[PHOT-107] Local-only Ollama swarm entrypoint."""

from __future__ import annotations

import argparse
import os

from src.constants import DEFAULT_LOCAL_OLLAMA_MODEL
from src.swarm import run_swarm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local-only Ollama swarm over a photonics inverse-design problem")
    parser.add_argument("--spec", required=True, help="Path to the problem spec JSON file")
    parser.add_argument("--model", default=DEFAULT_LOCAL_OLLAMA_MODEL, help="Local Ollama model to use for every agent")
    parser.add_argument("--rounds", type=int, default=1, help="Number of swarm-guided optimization rounds after baseline")
    parser.add_argument("--no-refine", action="store_true", help="Skip solver-backed refinement at the end of each round")
    return parser


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = build_parser().parse_args()
    run_swarm(args.spec, model=args.model, rounds=args.rounds, refine=not args.no_refine)
