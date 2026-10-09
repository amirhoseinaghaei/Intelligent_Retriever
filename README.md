# Retrieval RL learning project

Two experiments search the same small document collection: `train.py` uses Ray
RLlib PPO to choose a search strategy; `train_llm.py` uses TRL GRPO to fine-tune
an LLM's LoRA adapter to generate search queries.

## Run

Use Python 3.11 in a virtual environment. Install PyTorch's CPU build for this
first exercise if you want to avoid downloading CUDA packages.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python retrieval.py
python train.py --iterations 30
```

The collection contains six fictional operations documents. Training uses twelve
questions; evaluation uses six held-out paraphrases. This is an educational smoke
test, not evidence of generalization to real resources. The alias dictionary is
handwritten and shared by every baseline and the policy.

## Understand the experiment

- Observation: question word indicators and literal retrieval scores.
- Action: literal search, alias expansion, or alias expansion with stopword removal.
- Episode: one question and one retrieval action; this is a contextual bandit.
- Reward: reciprocal rank of the labeled relevant document in the top two results,
  minus 0.03 for expansion. Gold labels are never part of the observation.
- Evaluation: recall@2 and MRR@2, compared with all three fixed strategies printed
  by `retrieval.py`. A learned policy may fail to beat always using expansion.

Inspect `retrieval.py` first, then `RetrievalEnv` in `train.py`, then PPOConfig.
Checkpoints go to `checkpoints/ppo`. Repeat runs with different seeds and enlarge
the evaluation set before claiming improvements.

## LLM query generation and LoRA RL training

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. TRL handles autoregressive
generation and GRPO; PEFT freezes the base weights and trains rank-8 LoRA adapters
on the attention query and value projections. This script does not start Ray.
GRPO compares rewards within each question's sampled query group and optimizes
the generated tokens, with a small KL penalty against the frozen base model.

```bash
source .venv/bin/activate
python -m pip install -r requirements-llm.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python train_llm.py --steps 1 --output outputs/llm-smoke
python train_llm.py --steps 30 --samples 2
python query_llm.py "Can I recover an archive?"
```

Run the one-step command first to check generation and backward propagation.
On Ubuntu, install the C build tools needed by Triton's CUDA helpers first:

```bash
sudo apt update
sudo apt install -y build-essential
gcc --version
```

The first run downloads the model from Hugging Face. CUDA-capable PyTorch and a
working NVIDIA driver are needed for GPU training. If CUDA is unavailable, the
training script exits with instructions; `--cpu` explicitly enables a slow CPU
run. Outputs must use a new or empty directory. To query the smoke-run adapter,
pass `--adapter outputs/llm-smoke/adapter`.

Each training question produces two sampled queries by default. Queries are
searched literally (`action=0`); handwritten alias expansion is not used. Prompts
contain the question and general knowledge-base topics, but no gold document ID
or gold document text. Gold labels are passed only to the reward function.

The reward is reciprocal rank within the top two positive-overlap results:
first = 1, second = 0.5, absent = 0. Empty queries, multiline responses, or queries
longer than 12 words earn -1. Filtering zero-overlap hits prevents rewarding
alphabetical ties for failed searches. Positive-score ties still use the starter's
alphabetical tie-break. This simple reward can still favor keyword stuffing or
collection-specific memorization; it is a learning exercise rather than a robust
production retrieval objective.

The starting configuration uses 32 output tokens, one training sequence per
microbatch, two accumulated microbatches, gradient checkpointing, and BF16 on
supported GPUs. Generation holds two samples together. There is no vLLM server
or separate reward model. These settings are intended as a small starting point
for your RTX 4080 and 16 GB system RAM; actual memory use requires measurement.

Artifacts in `outputs/llm-grpo`:

- `rollouts.jsonl`: sampled training queries, rankings, and rewards.
- `before.json` and `after.json`: deterministic generation on held-out questions.
- `summary.json`: before/after metrics and whether adapter weights changed.
- `adapter/`: saved LoRA weights and tokenizer, loadable by `query_llm.py`.

When all queries in a group earn the same reward, the retrieval advantages are
zero. Inspect the rollouts and TRL's reward-variance metrics; more steps alone
cannot fix a missing reward signal. Try `--samples 4` or improve the dataset.
Evaluation may stay unchanged or get worse; no improvement is assumed.

The six documents and held-out paraphrases are a smoke test. For a useful next
experiment, add real resources and relevant-chunk labels, hold out documents and
topics, and compare against literal search and fixed expansion baselines. A
supervised query-rewriting warm-up can help if the small model struggles to
produce valid queries.

## Validation

```bash
python -m unittest test_llm_task.py
python -m py_compile llm_task.py train_llm.py query_llm.py
```

Reward checks and syntax validation run without downloading a model. GPU access
and dependency downloads were unavailable in the authoring environment, so the
full GRPO generation/backward pass and adapter reload need local validation with
the one-step run above. Dependencies are pinned to the versioned TRL API below.

References: [RLlib documentation](https://docs.ray.io/en/latest/rllib/index.html)
and [TRL GRPO 0.23.1](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer).
