"""[PHOT-104][PHOT-105] Train or run inverse-design workflows."""

from __future__ import annotations

import argparse

from src.workflows import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run surrogate training or inverse design")
    parser.add_argument("--spec", required=True, help="Path to the problem spec JSON file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("surrogate", "inverse", "refine"),
        help="Runtime mode to execute",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_train(args.spec, args.mode)
