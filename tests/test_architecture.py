"""Check that preparation and CLI boundaries do not load model dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_preparation_does_not_import_training_dependencies(self):
        code = """
import sys
from rl_posttraining.runtime.snapshot_data import load_dataset, prepare_dataset
from rl_posttraining.runtime.experiment import run
assert not any(name in sys.modules for name in ('ray', 'torch', 'transformers', 'peft'))
"""
        subprocess.run([sys.executable, "-c", code], check=True, cwd="/tmp")

    def test_supported_commands_show_help_without_credentials(self):
        environment = os.environ.copy()
        environment.pop("OPENROUTER_API_KEY", None)
        environment.pop("OPENROUTER_JUDGE_MODEL", None)
        for command in ("train_llm", "checkpoints", "extract"):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", f"rl_posttraining.cli.{command}", "--help"],
                    check=True, capture_output=True, text=True, env=environment, cwd="/tmp")
                self.assertIn("usage:", result.stdout)

    def test_standalone_entry_points(self):
        for script in ("train_llm.py", "scripts/checkpoint_manager.py", "scripts/extract_investigate_data.py"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                        check=True, capture_output=True, text=True, cwd="/tmp")
                self.assertIn("usage:", result.stdout)

    def test_training_requires_an_explicit_snapshot_source(self):
        result = subprocess.run([sys.executable, str(ROOT / "train_llm.py")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--experiments --prepared", result.stderr)
