"""Check Ray adapters and shutdown behavior without starting Ray processes."""
import sys
import os
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rl_posttraining.integrations import ray_training


class Remote:
    def __init__(self, function):
        self.remote = function


class RayTrainingTests(unittest.TestCase):
    def test_localhost_remains_loopback_before_ray_starts(self):
        code = """
import sys
from types import SimpleNamespace
from unittest.mock import patch
from rl_posttraining.integrations.ray_training import training_session
assert "ray" not in sys.modules
# Fail the CUDA check after importing Ray but before starting any processes.
no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
with patch.dict(sys.modules, {"torch": no_cuda}):
    try:
        with training_session({"cpu": False}, "test-key"):
            raise AssertionError("Should fail the CUDA check")
    except RuntimeError as error:
        assert "CUDA is unavailable" in str(error)
from ray._private.services import resolve_ip_for_localhost
assert resolve_ip_for_localhost("127.0.0.1") == "127.0.0.1"
"""
        environment = os.environ.copy()
        environment.pop("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", None)
        subprocess.run([sys.executable, "-c", code], env=environment, check=True)

    def test_one_actor_is_reused_and_always_released(self):
        actor = SimpleNamespace(sample=Remote(lambda question: [question, "other"]),
                                update=Remote(lambda rewards: {"rewards": rewards}),
                                save=Remote(lambda path: True))
        factory = Mock(return_value=actor)
        ray = SimpleNamespace(init=Mock(), remote=lambda **kwargs: Remote,
                              get=lambda value: value, kill=Mock(), shutdown=Mock())
        config = {"cpu": True, "judge_model": "judge", "artifactnet_timeout": 180}
        with patch.dict(sys.modules, {"ray": ray, "torch": SimpleNamespace()}), \
                patch.object(ray_training, "LearnerActor", factory), \
                patch.object(ray_training, "score_investigate", return_value={"reward": 0.8}):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with ray_training.training_session(config, "secret") as (learner, scorer):
                    for _ in range(2):
                        queries = learner.sample("question")
                        results = scorer.score(queries, {}, "endpoint")
                        self.assertEqual(learner.update([r["reward"] for r in results]), {"rewards": [0.8, 0.8]})
                    self.assertTrue(learner.save("adapter"))
                    raise RuntimeError("training failed")
        factory.assert_called_once_with(config)
        ray.kill.assert_called_once_with(actor)
        ray.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
