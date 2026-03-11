"""[PHOT-103] Prepare materials or datasets for photonics inverse design."""

from __future__ import annotations

import argparse

from autodesign.workflows import run_prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached photonics materials or datasets")
    parser.add_argument("--spec", required=True, help="Path to the problem spec JSON file")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("materials", "dataset"),
        help="Preparation stage to execute",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_prepare(args.spec, args.stage)
