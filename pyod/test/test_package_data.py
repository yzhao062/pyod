# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class TestPackageData(unittest.TestCase):
    def test_model_analysis_jsons_are_in_sdist(self):
        repo_root = Path(__file__).resolve().parents[2]
        source_dir = repo_root / "pyod" / "utils" / "model_analysis_jsons"
        self.assertTrue(source_dir.exists())
        self.assertTrue(any(source_dir.glob("*.json")))

        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir) / "dist"
            dist_dir.mkdir(parents=True, exist_ok=True)

            subprocess.check_call(
                [sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)],
                cwd=str(repo_root),
            )

            archives = sorted(dist_dir.glob("*.tar.gz"))
            self.assertTrue(archives, "sdist archive was not created")

            with tarfile.open(archives[0], "r:gz") as archive:
                members = archive.getnames()

            has_model_json = any(
                member.endswith(".json")
                and os.path.normpath(member).replace("\\", "/").find(
                    "/pyod/utils/model_analysis_jsons/"
                )
                != -1
                for member in members
            )
            self.assertTrue(
                has_model_json,
                "model_analysis_jsons JSON files are missing from sdist package",
            )
