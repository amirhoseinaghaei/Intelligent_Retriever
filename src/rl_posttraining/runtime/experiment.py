"""Compose preparation, training infrastructure, and output for one experiment."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from rl_posttraining.integrations.environments import CheckpointEnvironments
from rl_posttraining.integrations.checkpoints import check_docker
from rl_posttraining.integrations.files import write_json
from rl_posttraining.integrations.ray_training import training_session
from .artifacts import TrainingLog
from .snapshot_data import load_dataset, prepare_dataset
from .training import train_snapshots


@contextmanager
def provider_credentials(api_key):
    """Make the judge key available to Compose without saving it to run files."""
    previous = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = previous


def run(args, reward_config=None):
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is nonempty; choose another --output")
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = (args.prepared.expanduser().resolve() if args.prepared else
                prepare_dataset(args.experiments, args.output / "prepared", args))
    dataset = load_dataset(prepared)
    if args.prepare_only:
        print(f"Prepared dataset: {prepared}\nUse --prepared {prepared} with a new --output to train.")
        return
    if reward_config is None:
        raise ValueError("Training requires reward configuration")
    check_docker()
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["prepared"] = str(prepared)
    write_json(args.output / "config.json", config)
    environments = CheckpointEnvironments(prepared / "environments", args)
    with provider_credentials(reward_config.api_key), \
            training_session(config, reward_config.api_key) as (learner, scorer), \
            TrainingLog(args.output) as log:
        summary = train_snapshots(args, prepared, dataset, learner, scorer, environments, log)
    print(json.dumps(summary, indent=2))
