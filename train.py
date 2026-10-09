"""RLlib PPO learns a one-step retrieval strategy (a contextual bandit)."""
import argparse
import os
from pathlib import Path

# Ray 2.59 otherwise rewrites loopback to an automatically selected LAN/VPN IP.
# This internal setting disables multi-node operation for this local exercise.
# It must be set before importing Ray (including indirectly through RLlib).
os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "0"

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns

from retrieval import ACTIONS, DOCUMENTS, TEST, TRAIN, retrieve, tokens
from retrieval import reward as retrieval_reward

VOCAB = sorted({t for q, _ in TRAIN for t in tokens(q)})


def encode(question):
    words = set(tokens(question))
    # Literal-search scores help the policy decide whether to reformulate.
    _, scores = retrieve(question, 0)
    return np.array([float(t in words) for t in VOCAB] + list(scores.values()), dtype=np.float32)


class RetrievalEnv(gym.Env):
    def __init__(self, config=None):
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 1, (len(VOCAB) + len(DOCUMENTS),), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.question, self.gold = TRAIN[int(self.np_random.integers(len(TRAIN)))]
        return encode(self.question), {}

    def step(self, action):
        value = retrieval_reward(self.question, self.gold, int(action))
        return encode(self.question), value, True, False, {}


def evaluate(module):
    ranks = []
    module.eval()
    with torch.no_grad():
        for question, _ in TEST:
            obs = torch.from_numpy(encode(question)).unsqueeze(0)
            output = module.forward_inference({Columns.OBS: obs})
            action = int(output[Columns.ACTION_DIST_INPUTS].argmax(-1).item())
            ranked, _ = retrieve(question, action)
            ranks.append(ranked)
            print(f"{ACTIONS[action]} | {question} | {ranked}")
    recall = sum(gold in rank for rank, (_, gold) in zip(ranks, TEST)) / len(TEST)
    mrr = sum(1 / (rank.index(gold) + 1) if gold in rank else 0
              for rank, (_, gold) in zip(ranks, TEST)) / len(TEST)
    print(f"Held-out recall@2={recall:.3f}, MRR@2={mrr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    # This single-machine exercise should use loopback rather than a VPN/LAN IP.
    ray.init(address="local", _node_ip_address="127.0.0.1",
             include_dashboard=False, num_cpus=2,
             object_store_memory=128 * 1024 * 1024)
    algo = None
    try:
        config = (PPOConfig().environment(RetrievalEnv).framework("torch")
                  .env_runners(num_env_runners=0, rollout_fragment_length=64)
                  .learners(num_learners=0, num_gpus_per_learner=0)
                  .training(train_batch_size_per_learner=256, minibatch_size=64,
                            num_epochs=4, lr=0.001, gamma=0.0)
                  .debugging(seed=42))
        algo = config.build_algo()
        for i in range(args.iterations):
            result = algo.train()
            metrics = result.get("env_runners", {})
            print(f"iteration={i + 1}, mean_reward={metrics.get('episode_return_mean')}")
        evaluate(algo.get_module())
        print("Checkpoint:", algo.save_to_path(str(Path("checkpoints/ppo").resolve())))
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
