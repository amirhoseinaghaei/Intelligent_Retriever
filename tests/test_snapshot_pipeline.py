"""Exercise preparation, exact routing, persistent learning, and teardown offline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from rl_posttraining.cli.train_llm import parse_args
from rl_posttraining.config import RewardConfig
from rl_posttraining.integrations import checkpoints, snapshots
from rl_posttraining.runtime.snapshot_data import load_dataset, prepare_dataset
from rl_posttraining.runtime.training import train_snapshots
from rl_posttraining.runtime.artifacts import TrainingLog
from rl_posttraining.integrations.environments import CheckpointEnvironments
from rl_posttraining.runtime import experiment
from contextlib import contextmanager


class Worker:
    def __init__(self):
        self.steps = 0
        self.questions = []

    def sample(self, question):
        self.questions.append(question)
        return ["useful query", "irrelevant query"]

    def update(self, rewards):
        self.steps += 1
        return {"advantages": [1, -1], "mean_reward": sum(rewards) / 2, "loss": 0.1}

    def save(self, path):
        return self.steps > 0


class SnapshotPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "experiments"
        self.prepared = self.root / "prepared"
        self.output = self.root / "training"
        self.output.mkdir()
        self.args = parse_args(["--experiments", str(self.source), "--snapshot-epochs", "2",
                                "--steps-per-snapshot", "2", "--output", str(self.output)])
        self.args.judge_model = "judge"
        for run in ("run-a", "run-b", "run-without-snapshot"):
            self.make_archive(run, snapshot=run != "run-without-snapshot")

    def make_archive(self, run, snapshot):
        archive = self.source / "problem" / run / "checkpoint_1/experiment.zip"
        archive.parent.mkdir(parents=True)
        session = {"session_id": "same-session-id", "messages": [{"parts": [{
            "type": "tool", "tool": "MCP_ARTNET_artifactnet_investigate", "callID": "same-call-id",
            "state": {"status": "completed", "input": {"changeRequest": f"Request for {run}",
                        "entryNodes": "Requirement", "kTop": 5, "hops": 1, "threshold": 0.5},
                      "output": "REFERENCE_OUTPUT_MUST_NOT_ENTER_PROMPT"}}]}]}
        files = {"checkout/.agent_workspace/agents/worker-1/sessions/s.json": json.dumps(session).encode()}
        if snapshot:
            files.update({"services/neo4j.dump": run.encode(),
                          snapshots.ARTNET + "schema.yaml": b"version: test",
                          snapshots.ARTNET + "config.yaml": b"schema: .artnet/schema.yaml"})
        manifest = {"files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                    "images": {"neo4j": {"id": "neo4j:test"},
                               "artifactnet-backend": {"id": "artifactnet:test"}}}
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("manifest.json", json.dumps(manifest))
            for name, data in files.items():
                zipped.writestr(name, data)

    def prepare(self):
        with patch("builtins.print"):
            prepare_dataset(self.source, self.prepared, self.args)
        return load_dataset(self.prepared)

    def test_prepare_separates_runs_and_skips_missing_snapshots(self):
        dataset = self.prepare()
        self.assertEqual(dataset["examples"], 2)
        self.assertEqual(len(dataset["snapshots"]), 2)
        self.assertEqual(len(dataset["skipped"]), 1)
        for snapshot in dataset["snapshots"]:
            folder = self.prepared / "environments" / snapshot["id"]
            record, = json.loads((folder / "training_data.json").read_text())
            self.assertEqual(record["metadata"]["archive"], snapshot["name"])
            self.assertEqual(record["y"], "REFERENCE_OUTPUT_MUST_NOT_ENTER_PROMPT")
            self.assertTrue((folder / "backup/neo4j.dump").exists())

    def test_same_learner_trains_only_on_active_snapshot_then_removes_it(self):
        dataset = self.prepare()
        snapshots = {s["id"]: s for s in dataset["snapshots"]}
        active = []
        events = []
        worker = Worker()

        def start(state, identity, args):
            self.assertFalse(active)
            active.append(identity)
            events.append(("start", identity, worker.steps))
            return {"backend_port": 8001}

        def stop(state, identity, remove_data):
            self.assertTrue(remove_data)
            self.assertEqual(active.pop(), identity)
            events.append(("stop", identity, worker.steps))

        def score(query, record, endpoint, model, timeout):
            self.assertEqual(record["metadata"]["archive"], snapshots[active[0]]["name"])
            self.assertEqual(endpoint, "http://127.0.0.1:8001/artifactnet")
            return {"query": query, "reward": float(query == "useful query")}

        with patch.object(checkpoints, "start", side_effect=start), \
                patch.object(checkpoints, "stop", side_effect=stop), patch("builtins.print"):
            scorer = SimpleNamespace(score=lambda queries, record, endpoint:
                                     [score(query, record, endpoint, "judge", 180) for query in queries])
            environments = CheckpointEnvironments(self.prepared / "environments", self.args)
            with TrainingLog(self.output) as log:
                summary = train_snapshots(self.args, self.prepared, dataset, worker, scorer, environments, log)
        self.assertEqual(worker.steps, 8)
        self.assertEqual(summary["steps"], 8)
        self.assertFalse(active)
        self.assertEqual([event[0] for event in events], ["start", "stop"] * 4)
        self.assertEqual([step for event, _, step in events if event == "start"], [0, 2, 4, 6])
        self.assertTrue(all(question.startswith("Request for run-") for question in worker.questions))
        rows = [json.loads(line) for line in (self.output / "rollouts.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["source"]["archive"], snapshots[row["snapshot_id"]]["name"])

    def test_cleanup_on_startup_and_training_failure(self):
        for failure_on_start in (False, True):
            with self.subTest(failure_on_start=failure_on_start), \
                    patch.object(checkpoints, "start", side_effect=RuntimeError("startup") if failure_on_start else None,
                                 return_value={"backend_port": 8001}), \
                    patch.object(checkpoints, "stop") as stop:
                with self.assertRaisesRegex(RuntimeError, "startup|training"):
                    with CheckpointEnvironments(self.root, self.args).open("identity"):
                        raise RuntimeError("training")
                stop.assert_called_once_with(self.root, "identity", remove_data=True)

    def test_wrong_archive_is_rejected_before_training(self):
        dataset = self.prepare()
        path = self.prepared / "environments" / dataset["snapshots"][0]["id"] / "training_data.json"
        records = json.loads(path.read_text())
        records[0]["metadata"]["archive"] = "other/run-a/checkpoint_1/experiment.zip"
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "wrong snapshot"):
            load_dataset(self.prepared)

    def test_modified_dump_is_rejected_before_training(self):
        dataset = self.prepare()
        path = self.prepared / "environments" / dataset["snapshots"][0]["id"] / "backup/neo4j.dump"
        path.write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_dataset(self.prepared)

    def test_prepare_only_cli_needs_no_reward_config_or_api_key(self):
        environment = os.environ.copy()
        environment.pop("OPENROUTER_API_KEY", None)
        environment.pop("OPENAI_API_KEY", None)
        script = Path(__file__).resolve().parents[1] / "train_llm.py"
        subprocess.run([sys.executable, str(script), "--experiments", str(self.source),
                        "--prepare-only", "--config", str(self.root / "missing-config.toml"),
                        "--output", str(self.output)], env=environment, check=True, capture_output=True)
        self.assertTrue((self.output / "prepared/dataset.json").exists())

    def test_experiment_composes_services_and_never_saves_credentials(self):
        self.prepare()
        self.args.prepared, self.args.experiments = self.prepared, None
        worker = Worker()
        scorer = SimpleNamespace(score=lambda queries, record, endpoint:
                                 [{"query": q, "reward": 0.5} for q in queries])

        @contextmanager
        def session(config, key):
            self.assertEqual(key, "training-secret")
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "training-secret")
            yield worker, scorer

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "previous-key"}), \
                patch.object(experiment, "check_docker"), \
                patch.object(experiment, "training_session", side_effect=session) as training, \
                patch.object(checkpoints, "start", return_value={"backend_port": 8001}), \
                patch.object(checkpoints, "stop") as stop, patch("builtins.print"):
            experiment.run(self.args, RewardConfig(api_key="training-secret", model="judge"))
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "previous-key")
        training.assert_called_once()
        self.assertEqual(stop.call_count, 4)
        self.assertEqual(worker.steps, 8)
        self.assertNotIn("training-secret", (self.output / "config.json").read_text())

    def test_docker_failure_is_reported_before_starting_ray(self):
        self.prepare()
        self.args.prepared, self.args.experiments = self.prepared, None
        with patch.object(experiment, "check_docker", side_effect=RuntimeError("Docker socket permission denied")), \
                patch.object(experiment, "training_session") as training:
            with self.assertRaisesRegex(RuntimeError, "Docker socket permission denied"):
                experiment.run(self.args, RewardConfig(api_key="secret", model="judge"))
        training.assert_not_called()
        self.assertFalse((self.output / "config.json").exists())

    def test_failed_scoring_preserves_candidates_without_updating_model(self):
        dataset = self.prepare()
        learner = Worker()
        scorer = SimpleNamespace(score=lambda *args: (_ for _ in ()).throw(RuntimeError('ArtifactNet HTTP 400: explanation')))
        environments = CheckpointEnvironments(self.prepared / 'environments', self.args)
        with patch.object(checkpoints, 'start', return_value={'backend_port': 8001}), \
                patch.object(checkpoints, 'stop') as stop, patch('builtins.print'):
            with TrainingLog(self.output) as log:
                with self.assertRaisesRegex(RuntimeError, 'HTTP 400: explanation'):
                    train_snapshots(self.args, self.prepared, dataset, learner, scorer, environments, log)
        self.assertEqual(learner.steps, 0)
        stop.assert_called_once()
        request = json.loads((self.output / 'requests.jsonl').read_text())
        failure = json.loads((self.output / 'failures.jsonl').read_text())
        self.assertEqual(request['queries'], ['useful query', 'irrelevant query'])
        self.assertEqual(request['step'], failure['step'])
        self.assertIn('HTTP 400: explanation', failure['error'])


if __name__ == "__main__":
    unittest.main()
