from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.swarm import ScriptedOllamaClient, run_swarm

from tests.support import write_spec


class LocalSwarmTests(unittest.TestCase):
    def test_scripted_swarm_runs_local_design_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec_path = write_spec(
                root,
                overrides={
                    "problem_id": "swarm_problem",
                    "dataset": {"num_samples": 8},
                    "surrogate": {"epochs": 2, "hidden_channels": 8},
                    "inverse_design": {"steps": 6, "top_k": 2},
                },
            )
            client = ScriptedOllamaClient(
                {
                    "photonics_strategist": [
                        {
                            "summary": "Expand the material search modestly.",
                            "changes": {
                                "dataset.num_samples": 10,
                                "material_search_policy.max_candidates": 4,
                            },
                        }
                    ],
                    "surrogate_tuner": [
                        {
                            "summary": "Train slightly longer with a wider model.",
                            "changes": {
                                "surrogate.epochs": 3,
                                "surrogate.hidden_channels": 12,
                            },
                        }
                    ],
                    "inverse_tuner": [
                        {
                            "summary": "Search a bit harder in inverse mode.",
                            "changes": {
                                "inverse_design.steps": 8,
                                "inverse_design.learning_rate": 0.06,
                            },
                        }
                    ],
                    "reviewer": [
                        {
                            "summary": "Keep the stronger training and inverse-search changes.",
                            "changes": {
                                "dataset.num_samples": 10,
                                "surrogate.epochs": 3,
                                "inverse_design.steps": 8,
                            },
                        }
                    ],
                }
            )
            payload = run_swarm(spec_path, model="gemma3:4b", rounds=1, client=client, refine=True)
            self.assertEqual(payload["best_round"], min(payload["history"], key=lambda item: item["summary"]["best_candidate_loss"])["round"])
            self.assertEqual(len(payload["history"]), 2)
            self.assertTrue((root / "runs" / "swarm_problem" / "swarm" / "swarm_log.json").exists())
            self.assertEqual(payload["history"][1]["changes"]["surrogate.epochs"], 3)


if __name__ == "__main__":
    unittest.main()
