"""Load reward settings once in the driver before starting Ray."""
from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RewardConfig:
    api_key: str = field(repr=False)
    model: str


def load_reward_config(path=None, *, model=None):
    config_path = Path(path) if path is not None else Path("config.toml")
    try:
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError:
        if path is not None:
            raise ValueError(f"Configuration file not found: {config_path}") from None
        data = {}
    except (OSError, tomllib.TOMLDecodeError):
        raise ValueError(f"Cannot read valid TOML configuration: {config_path}") from None
    reward = data.get("reward", {})
    if not isinstance(reward, dict):
        raise ValueError("config.toml must contain a [reward] table")
    if reward.get("provider", "openrouter") != "openrouter":
        raise ValueError("The reward calculator currently supports provider = 'openrouter'")
    api_key = os.environ.get("OPENROUTER_API_KEY") or reward.get("api_key")
    judge_model = model or os.environ.get("OPENROUTER_JUDGE_MODEL") or reward.get("model")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Set api_key in config.toml [reward] or OPENROUTER_API_KEY")
    if not isinstance(judge_model, str) or not judge_model.strip():
        raise ValueError("Set model in config.toml [reward], OPENROUTER_JUDGE_MODEL, or --judge-model")
    return RewardConfig(api_key=api_key.strip(), model=judge_model.strip())
