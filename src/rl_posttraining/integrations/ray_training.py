"""Translate synchronous training interfaces into Ray actor and task calls."""
from contextlib import contextmanager
import os

from .artifactnet import score_investigate
from .models import load_learner


class LearnerActor:
    """The single actor retains its model and optimizer for the entire run."""
    def __init__(self, config):
        self.learner = load_learner(config)
        self.samples = config["samples"]
        self.max_tokens = config["max_query_tokens"]

    def sample(self, question):
        return self.learner.sample(question, self.samples, self.max_tokens)

    def update(self, rewards):
        return self.learner.update(rewards)

    def save(self, path):
        return self.learner.save(path)


class RayLearner:
    def __init__(self, ray, actor):
        self.ray, self.actor = ray, actor

    def sample(self, question):
        return self.ray.get(self.actor.sample.remote(question))

    def update(self, rewards):
        return self.ray.get(self.actor.update.remote(rewards))

    def save(self, path):
        return self.ray.get(self.actor.save.remote(path))


class RayScorer:
    def __init__(self, ray, judge_model, timeout):
        self.ray = ray
        self.task = ray.remote(num_cpus=1)(score_investigate)
        self.judge_model, self.timeout = judge_model, timeout

    def score(self, queries, record, endpoint):
        return self.ray.get([self.task.remote(query, record, endpoint, self.judge_model, self.timeout)
                             for query in queries])


@contextmanager
def training_session(config, api_key):
    """Start Ray once and always release the actor and local Ray processes."""
    # Ray 2.59 otherwise rewrites 127.0.0.1 to an auto-detected interface.
    # This experiment uses one machine; keep coordinator and workers on loopback.
    # Ray reads the flag at import time, so it must precede `import ray`.
    os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "0"
    try:
        import ray
        import torch
    except ImportError as error:
        raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from error
    if not config["cpu"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check NVIDIA/PyTorch or explicitly use --cpu.")
    actor = None
    try:
        ray.init(address="local", _node_ip_address="127.0.0.1", include_dashboard=False,
                 num_cpus=2, num_gpus=0 if config["cpu"] else 1,
                 object_store_memory=128 * 1024 * 1024,
                 runtime_env={"env_vars": {"OPENROUTER_API_KEY": api_key}})
        actor_class = ray.remote(num_cpus=1, num_gpus=0 if config["cpu"] else 1)(LearnerActor)
        actor = actor_class.remote(config)
        yield RayLearner(ray, actor), RayScorer(ray, config["judge_model"], config["artifactnet_timeout"])
    finally:
        try:
            if actor is not None:
                ray.kill(actor)
        finally:
            ray.shutdown()
