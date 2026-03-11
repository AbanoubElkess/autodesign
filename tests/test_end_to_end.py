from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autodesign.dataset import load_dataset
from autodesign.workflows import run_prepare, run_train

from tests.support import write_spec


class EndToEndWorkflowTests(unittest.TestCase):
    def test_mock_workflow_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec_path = write_spec(root)
            run_prepare(spec_path, "dataset")
            run_train(spec_path, "surrogate")
            run_train(spec_path, "inverse")
            before_refine = load_dataset(root / "cache" / "test_problem" / "dataset.pt")
            run_train(spec_path, "refine")
            after_refine = load_dataset(root / "cache" / "test_problem" / "dataset.pt")
            self.assertTrue((root / "runs" / "test_problem" / "surrogate.pt").exists())
            self.assertTrue((root / "runs" / "test_problem" / "candidates.json").exists())
            self.assertGreater(len(after_refine["records"]), len(before_refine["records"]))


if __name__ == "__main__":
    unittest.main()
