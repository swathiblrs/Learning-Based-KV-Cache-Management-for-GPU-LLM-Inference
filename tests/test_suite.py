import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from kv_cache.suite import run_suite


class SuiteTests(unittest.TestCase):
    def test_failure_preserves_diagnostics_archive(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            with patch("kv_cache.suite.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["test"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_suite(output, requests=2, repeats=1)
            with zipfile.ZipFile(str(output) + ".zip") as z:
                manifest = json.loads(z.read("manifest.json"))
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["completed"], [])

    def test_existing_results_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                run_suite(d)
