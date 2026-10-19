"""Reward configuration loading and credential boundaries."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_posttraining.config import load_reward_config
from rl_posttraining.cli.train_llm import main


class RewardConfigTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("rl_posttraining.runtime.experiment.run")
    def test_training_passes_file_credentials_separately_from_saved_arguments(self, run):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[reward]\napi_key = "file-secret"\nmodel = "provider/judge"\n')
            with patch("sys.argv", ["train_llm.py", "--prepared", "prepared-data", "--config", str(path), "--cpu"]):
                main()
        args, reward_config = run.call_args.args
        self.assertEqual(args.judge_model, "provider/judge")
        self.assertEqual(reward_config.api_key, "file-secret")
        self.assertNotIn("file-secret", json.dumps(vars(args), default=str))

    @patch.dict(os.environ, {}, clear=True)
    def test_file_settings_and_secret_repr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[reward]\nprovider = "openrouter"\napi_key = "file-secret"\nmodel = "provider/judge"\n')
            config = load_reward_config(path)
        self.assertEqual(config.api_key, "file-secret")
        self.assertEqual(config.model, "provider/judge")
        self.assertNotIn("file-secret", repr(config))

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-secret",
                            "OPENROUTER_JUDGE_MODEL": "env/judge"}, clear=True)
    def test_environment_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[reward]\napi_key = "file-secret"\nmodel = "file/judge"\n')
            self.assertEqual(load_reward_config(path).api_key, "env-secret")
            self.assertEqual(load_reward_config(path).model, "env/judge")
            self.assertEqual(load_reward_config(path, model="cli/judge").model, "cli/judge")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_invalid_or_incomplete_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            with self.assertRaises(ValueError):
                load_reward_config(path)
            for content in ('[reward', '[reward]\napi_key = ""\nmodel = ""',
                            '[reward]\napi_key = 123\nmodel = "judge"',
                            '[reward]\nprovider = "unsupported"\napi_key = "secret"\nmodel = "judge"',
                            'reward = "invalid"'):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        load_reward_config(path)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-secret",
                            "OPENROUTER_JUDGE_MODEL": "env/judge"}, clear=True)
    def test_environment_only_when_default_file_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(load_reward_config().model, "env/judge")
            finally:
                os.chdir(previous)
