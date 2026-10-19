"""Train a query LoRA adapter using Ray and GRPO."""
import argparse
import subprocess
from pathlib import Path

from rl_posttraining.config import load_reward_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Reward configuration file (default: ./config.toml)")
    parser.add_argument("--judge-model", help="Override the configured reward model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiments", type=Path,
                        help="Extract examples/snapshots from this experiments folder, then train")
    source.add_argument("--prepared", type=Path,
                        help="Train from a previously extracted prepared dataset folder")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Extract and verify snapshot data without Docker, models, or API calls")
    parser.add_argument("--snapshot-epochs", type=int, default=1,
                        help="Passes over snapshots, shuffled each epoch")
    parser.add_argument("--steps-per-snapshot", type=int, default=5,
                        help="Sampled GRPO groups per snapshot per epoch")
    parser.add_argument("--neo4j-image", help="Compatible Neo4j image override when preparing snapshots")
    parser.add_argument("--artifactnet-backend-image", help="Compatible ArtifactNet image override when preparing snapshots")
    parser.add_argument("--port", type=int, default=8001, help="Local ArtifactNet port")
    parser.add_argument("--bolt-port", type=int, default=7687, help="Local Neo4j port")
    parser.add_argument("--snapshot-timeout", dest="timeout", type=int, default=240,
                        help="Seconds to wait for checkpoint services to become healthy")
    parser.add_argument("--artifactnet-timeout", type=int, default=180,
                        help="Timeout for each live investigate request")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-query-tokens", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=1,
                        help="Optimizer updates per group with fixed old probabilities")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("outputs/snapshot-grpo"))
    parser.add_argument("--cpu", action="store_true", help="Allow slow CPU training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.prepare_only and args.experiments is None:
        parser.error("--prepare-only requires --experiments")
    if min(args.snapshot_epochs, args.steps_per_snapshot, args.timeout, args.artifactnet_timeout) < 1:
        parser.error("Snapshot epochs, steps, and timeouts must be positive")
    if not (1 <= args.port <= 65535 and 1 <= args.bolt_port <= 65535) or args.port == args.bolt_port:
        parser.error("Use distinct ports between 1 and 65535")
    if args.samples < 2 or min(args.max_query_tokens, args.update_epochs) < 1:
        parser.error("samples must be >= 2; max-query-tokens and update-epochs must be positive")
    if args.learning_rate <= 0 or args.beta < 0 or not 0 < args.clip < 1:
        parser.error("learning-rate must be positive; beta >= 0; 0 < clip < 1")
    return args

def main():
    args = parse_args()
    from rl_posttraining.runtime.experiment import run
    try:
        reward_config = None
        if not args.prepare_only:
            reward_config = load_reward_config(args.config, model=args.judge_model)
            args.judge_model = reward_config.model
        run(args, reward_config)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
